import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


class WellProfiler:
    """
    Analizador de comportamiento de pozos de agua.
    Extrae ciclos operativos, estima parámetros hidrogeológicos (k, A, h', h_d),
    calcula estadísticas robustas y determina el régimen operativo.
    """

    def __init__(self, file_path=None, min_cycle_points=5, smooth_window=5):
        self.file_path = file_path
        self.min_cycle_points = min_cycle_points
        self.smooth_window = smooth_window
        self.df = None
        self.cycles_df = None
        self.stats = {}
        self.regime = "Desconocido"
        self.threshold = 0.5
        self.depth_like = False

    @classmethod
    def from_dataframe(cls, df, threshold, min_cycle_points=5, smooth_window=5):
        """
        Crea un WellProfiler a partir de un DataFrame ya cargado.
        El umbral debe calcularse sobre el dataset completo (mes entero) antes de llamar esto.
        """
        instance = cls(file_path=None, min_cycle_points=min_cycle_points, smooth_window=smooth_window)
        instance.threshold = threshold
        instance.df = df.copy()
        instance.df['pump_on'] = instance.df['caudal_ls'] > threshold
        instance._prepare_level_column()
        return instance

    @staticmethod
    def _detect_depth_like(nivel: np.ndarray, pump_on: np.ndarray) -> bool:
        """
        Detecta si nivel_m representa profundidad (sube cuando la bomba está ON)
        en lugar de nivel (baja cuando la bomba está ON).
        Requiere al menos 10 puntos en cada estado para ser confiable.
        """
        if pump_on.sum() < 10 or (~pump_on).sum() < 10:
            return False
        return bool(np.nanmedian(nivel[pump_on]) > np.nanmedian(nivel[~pump_on]))

    def _prepare_level_column(self):
        """
        Suaviza nivel_m con mediana rolling, detecta si es profundidad y crea
        la columna h (altura física, positiva hacia arriba) para que los ajustes
        exponenciales sean físicamente consistentes.
        """
        self.df['nivel_f'] = self.df['nivel_m'].rolling(
            window=self.smooth_window, center=True, min_periods=1
        ).median()
        self.depth_like = self._detect_depth_like(
            self.df['nivel_f'].to_numpy(float),
            self.df['pump_on'].to_numpy(bool),
        )
        self.df['h'] = -self.df['nivel_f'] if self.depth_like else self.df['nivel_f']

    @staticmethod
    def _fit_exp(t: np.ndarray, y: np.ndarray):
        """
        Ajusta y(t) = y_inf + (y[0] - y_inf) * exp(-(t - t[0]) / tau)
        con loss Huber (robusto a outliers). Parametriza tau en log-space
        para estabilidad numérica.
        Retorna: (y_inf, tau_s, rmse, r2, ok)
        """
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        valid = np.isfinite(t) & np.isfinite(y)
        t, y = t[valid], y[valid]
        if len(t) < 3:
            return np.nan, np.nan, np.nan, np.nan, False

        y0 = float(y[0])
        y_inf0 = float(np.nanmedian(y[-max(3, len(y) // 5):]))
        tau0 = max(float(t[-1] - t[0]) / 3.0, 1.0)
        x0 = np.array([y_inf0, np.log(tau0)])

        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        rng = abs(ymax - ymin) + 1e-6
        lb = np.array([ymin - 10 * rng, np.log(1e-3)])
        ub = np.array([ymax + 10 * rng, np.log(1e9)])

        def residuals(x):
            y_inf, logtau = x
            return y - (y_inf + (y0 - y_inf) * np.exp(-(t - t[0]) / np.exp(logtau)))

        try:
            res = least_squares(residuals, x0, bounds=(lb, ub), loss='huber', max_nfev=10000)
            y_inf_fit = float(res.x[0])
            tau_fit = float(np.exp(res.x[1]))
            r = residuals(res.x)
            rmse = float(np.sqrt(np.mean(r ** 2)))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = float(1 - np.sum(r ** 2) / ss_tot) if ss_tot > 0 else float('nan')
            return y_inf_fit, tau_fit, rmse, r2, bool(res.success)
        except Exception:
            return np.nan, np.nan, np.nan, np.nan, False

    def load_and_clean(self):
        """Carga datos, estandariza fechas, define umbral dinámico y prepara columna h."""
        try:
            self.df = pd.read_csv(self.file_path)
            self.df['ts'] = pd.to_datetime(self.df['ts'])
            self.df = self.df.sort_values('ts').reset_index(drop=True)

            # Umbral dinámico: mayor entre 0.5 l/s o el 5% del caudal máximo
            q_max = self.df['caudal_ls'].max()
            self.threshold = max(0.5, q_max * 0.05)

            self.df['pump_on'] = self.df['caudal_ls'] > self.threshold
            self._prepare_level_column()
            return True
        except Exception as e:
            print(f"Error cargando {self.file_path}: {e}")
            return False

    def extract_cycles(self):
        """
        Extrae ciclos operativos con debouncing. Por cada ciclo ON:
        - Ajusta exponencial a la fase OFF precedente → h' (nivel estático del acuífero)
        - Ajusta exponencial a la fase ON → h_d (nivel dinámico bajo bombeo)
        - Calcula k (conductividad hidráulica) y A (almacenamiento × área)
        - Integra numéricamente el volumen bombeado
        """
        df = self.df
        df['change'] = df['pump_on'].astype(int).diff()
        starts = df[df['change'] == 1].index

        cycles_data = []

        for start_idx in starts:
            ends = df[(df.index > start_idx) & (df['change'] == -1)].index
            if len(ends) == 0:
                continue
            end_idx = ends[0]

            on_window = df.loc[start_idx:end_idx - 1]
            if len(on_window) < self.min_cycle_points:
                continue

            q = on_window['caudal_ls']
            duration_min = (
                on_window['ts'].iloc[-1] - on_window['ts'].iloc[0]
            ).total_seconds() / 60.0

            # Volumen bombeado por integración numérica
            dt_arr = on_window['ts'].diff().dt.total_seconds().to_numpy()
            dt_arr[0] = float(np.nanmedian(dt_arr[1:])) if len(dt_arr) > 1 else 0.0
            vol_m3 = float(np.sum((q.to_numpy() / 1000.0) * dt_arr))

            # Ajuste exponencial fase ON → h_d (nivel dinámico bajo bombeo)
            t_on = (on_window['ts'] - on_window['ts'].iloc[0]).dt.total_seconds().to_numpy(float)
            h_d_fit, tau_on, rmse_on, r2_on, ok_on = self._fit_exp(t_on, on_window['h'].to_numpy(float))
            h_d_nivel = -h_d_fit if self.depth_like else h_d_fit

            # Segmento OFF previo → h' (nivel estático del acuífero)
            prev_offs = df[(df.index < start_idx) & (df['change'] == -1)].index
            off_start_idx = int(prev_offs[-1]) if len(prev_offs) > 0 else int(df.index[0])
            off_window = df.loc[off_start_idx:start_idx - 1]

            h_static_nivel = np.nan
            tau_off = np.nan
            rmse_off = np.nan
            r2_off = np.nan
            ok_off = False
            k = np.nan
            A = np.nan

            if len(off_window) >= self.min_cycle_points:
                t_off = (off_window['ts'] - off_window['ts'].iloc[0]).dt.total_seconds().to_numpy(float)
                h_static_fit, tau_off, rmse_off, r2_off, ok_off = self._fit_exp(
                    t_off, off_window['h'].to_numpy(float)
                )
                h_static_nivel = -h_static_fit if self.depth_like else h_static_fit

                # k = C / (h' - h_d), calculado en espacio h (físicamente consistente)
                q_m3s = float(q.median()) / 1000.0
                delta_h = h_static_fit - h_d_fit
                if np.isfinite(delta_h) and delta_h > 0 and q_m3s > 0:
                    k = float(q_m3s / delta_h)
                    tau_use = tau_off if (ok_off and np.isfinite(tau_off)) else tau_on
                    if np.isfinite(tau_use) and tau_use > 0:
                        A = float(tau_use * k)

            cycles_data.append({
                'start_ts': on_window['ts'].iloc[0],
                'duration_min': duration_min,
                'q_median': q.median(),
                'q_std': q.std(),
                'h_median': on_window['nivel_m'].median(),
                'h_static_m': h_static_nivel,
                'h_dinamico_m': h_d_nivel,
                'tau_off_s': tau_off,
                'tau_on_s': tau_on,
                'volumen_bombeado_m3': vol_m3,
                'k_m2_s': k,
                'A_m2': A,
                'rmse_off': rmse_off,
                'r2_off': r2_off,
                'rmse_on': rmse_on,
                'r2_on': r2_on,
                'ok_off': ok_off,
                'ok_on': ok_on,
            })

        self.cycles_df = pd.DataFrame(cycles_data)

    def compute_global_metrics(self):
        """Calcula métricas globales para alimentar el modelo de clasificación."""
        if self.df is None:
            return

        t_total_days = (self.df['ts'].max() - self.df['ts'].min()).total_seconds() / 86400

        if self.cycles_df is not None and not self.cycles_df.empty:
            total_cycles = len(self.cycles_df)
            freq_cycles_day = total_cycles / t_total_days
            avg_duration = self.cycles_df['duration_min'].mean()
            median_q = self.cycles_df['q_median'].median()
            variability_q = self.cycles_df['q_std'].mean()
            total_on_minutes = self.cycles_df['duration_min'].sum()
            duty_cycle_pct = (total_on_minutes / (t_total_days * 1440)) * 100

            vol_total_m3 = float(self.cycles_df['volumen_bombeado_m3'].sum())
            h_static_mean = float(np.nanmean(self.cycles_df['h_static_m']))
            h_d_mean = float(np.nanmean(self.cycles_df['h_dinamico_m']))
            k_mean = float(np.nanmean(self.cycles_df['k_m2_s']))
            A_mean = float(np.nanmean(self.cycles_df['A_m2']))
        else:
            # Caso especial: siempre ON (sin ciclos porque nunca bajó)
            if self.df['pump_on'].mean() > 0.95:
                duty_cycle_pct = 100.0
                freq_cycles_day = 0
                avg_duration = t_total_days * 1440
                median_q = self.df['caudal_ls'].median()
                variability_q = self.df['caudal_ls'].std()
            else:
                # Siempre OFF
                duty_cycle_pct = 0.0
                freq_cycles_day = 0
                avg_duration = 0
                median_q = 0
                variability_q = 0

            vol_total_m3 = 0.0
            h_static_mean = np.nan
            h_d_mean = np.nan
            k_mean = np.nan
            A_mean = np.nan

        def _safe_round(v, decimals):
            fv = float(v) if v is not None else float('nan')
            return round(fv, decimals) if np.isfinite(fv) else None

        self.stats = {
            'duty_cycle_pct': round(duty_cycle_pct, 2),
            'freq_cycles_day': round(freq_cycles_day, 2),
            'avg_cycle_duration_min': round(avg_duration, 1),
            'typical_flow_ls': round(median_q, 2),
            'flow_stability_std': round(variability_q, 2),
            'vol_total_m3': round(vol_total_m3, 3),
            'h_static_mean_m': _safe_round(h_static_mean, 3),
            'h_dinamico_mean_m': _safe_round(h_d_mean, 3),
            'k_mean_m2_s': _safe_round(k_mean, 6),
            'A_mean_m2': _safe_round(A_mean, 3),
            'nivel_es_profundidad': self.depth_like,
        }

    def classify_regime(self):
        """
        Determina la etiqueta de régimen basada en reglas heurísticas (Nettra).
        Estas reglas pueden ser luego refinadas por el Clustering HDBSCAN.
        """
        s = self.stats

        if s['duty_cycle_pct'] > 98.0:
            self.regime = "Siempre ON"
        elif s['duty_cycle_pct'] < 1.0:
            self.regime = "Siempre OFF"
        else:
            if s['freq_cycles_day'] > 10:
                self.regime = "Mixto Pulso"
            elif s['avg_cycle_duration_min'] > 360:
                self.regime = "Mixto Largo"
            else:
                self.regime = "Mixto Medio"

    def get_features(self):
        """Retorna un diccionario plano listo para un DataFrame de ML."""
        features = self.stats.copy()
        features['regime_label'] = self.regime
        features['dynamic_threshold_used'] = round(self.threshold, 2)
        return features

    def plot_summary(self):
        """Genera gráficos de validación: distribución de duración y niveles por ciclo."""
        if self.cycles_df is None or self.cycles_df.empty:
            print("No hay ciclos para graficar.")
            return

        has_physical = self.cycles_df['k_m2_s'].notna().any()

        if has_physical:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        else:
            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax2 = None

        ax1.hist(self.cycles_df['duration_min'], bins=30, color='skyblue', edgecolor='black')
        ax1.set_title(f"Distribución de Duración de Ciclos - {self.regime}")
        ax1.set_xlabel("Minutos")
        ax1.set_ylabel("Frecuencia")
        ax1.axvline(self.stats['avg_cycle_duration_min'], color='red', linestyle='--', label='Media')
        ax1.legend()

        if ax2 is not None:
            cidx = range(len(self.cycles_df))
            nivel_label = 'profundidad (m)' if self.depth_like else 'nivel (m)'
            ax2.scatter(cidx, self.cycles_df['h_static_m'], label="h' estático", color='steelblue', s=20)
            ax2.scatter(cidx, self.cycles_df['h_dinamico_m'], label="h_d dinámico", color='darkorange', s=20)
            ax2.set_title(f"Niveles por Ciclo ({nivel_label})")
            ax2.set_xlabel("Ciclo #")
            ax2.set_ylabel(nivel_label)
            ax2.legend()

        plt.tight_layout()
        plt.show()


# --- USO DEL SISTEMA ---

# 1. Instanciar
archivo = '../../Datos1/Agrícola Neumann SPA - Pozo N°2605 - 12ABA4_d6d8df20-1842-11ef-a52b-799a8dee664b.csv'
pozo = WellProfiler(archivo, min_cycle_points=5)

# 2. Ejecutar Pipeline
if pozo.load_and_clean():
    pozo.extract_cycles()
    pozo.compute_global_metrics()
    pozo.classify_regime()

    # 3. Obtener Resultados
    features = pozo.get_features()

    print("-" * 40)
    print(f"CLASIFICACIÓN AUTOMÁTICA: {features['regime_label']}")
    print("-" * 40)
    print(f"Umbral Dinámico Usado:         {features['dynamic_threshold_used']} l/s")
    print(f"Nivel es Profundidad:          {features['nivel_es_profundidad']}")
    print(f"Duty Cycle:                    {features['duty_cycle_pct']}%")
    print(f"Frecuencia:                    {features['freq_cycles_day']} ciclos/día")
    print(f"Caudal Típico (Mediana):       {features['typical_flow_ls']} l/s")
    print(f"Duración Media Ciclo:          {features['avg_cycle_duration_min']} min")
    print(f"Volumen Total Bombeado:        {features['vol_total_m3']} m³")
    print(f"Nivel Estático Medio (h'):     {features['h_static_mean_m']} m")
    print(f"Nivel Dinámico Medio (h_d):    {features['h_dinamico_mean_m']} m")
    print(f"Conductividad Hidráulica (k):  {features['k_mean_m2_s']} m²/s")
    print(f"Almacenamiento × Área (A):     {features['A_mean_m2']} m²")

    # 4. Validar visualmente
    pozo.plot_summary()
