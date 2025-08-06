# 🚀 Sistema de Trading con IA Multi-Capa V2

## 🎯 Descripción del Sistema

Sistema revolucionario de trading que combina estrategias tradicionales probadas (SMA8/50 + MACD) con inteligencia artificial de múltiples capas para trading en índices sintéticos. No reemplaza la lógica ganadora, sino que la potencia con:

- **Motor Multi-Estrategia**: 5 estrategias ejecutándose simultáneamente
- **IA de Ensemble**: 4 modelos especializados que predicen calidad, profit, duración y riesgo
- **Detección Automática de Regímenes**: Trending, Ranging, Breakout, Reversal
- **Gestión de Capital Inteligente**: Position sizing dinámico con Kelly Criterion
- **Aprendizaje Continuo**: Sistema que mejora sin overfitting

## 🏗️ Arquitectura del Sistema

### Capa 1: Motor de Señales Adaptativo
- Estrategia Original (SMA8/50 + MACD)
- Versión EMA alternativa
- SMAs Adaptativas (períodos basados en volatilidad)
- Confirmación Multi-Timeframe
- Estrategia Bollinger + RSI (backup)

### Capa 2: Filtros Inteligentes
- **Filtro de Régimen**: ADX, Choppiness Index, Fractales
- **Filtro de Calidad**: Divergencias, Volumen, S/R
- **Filtro Temporal**: Horarios óptimos, eventos importantes
- **Filtro de Contexto**: Correlaciones, Sentiment

### Capa 3: IA de Ensemble
- **Modelo Clasificación**: Calidad de señal
- **Modelo Regresión**: Profit esperado
- **Modelo Temporal**: Duración del movimiento
- **Modelo Riesgo**: Riesgo máximo probable

### Capa 4: Gestión de Capital
- Position sizing con Kelly Criterion modificado
- Stop loss adaptativos (ATR + volatilidad)
- Take profit dinámicos con trailing
- Gestión de correlaciones entre posiciones

## 📁 Estructura del Proyecto

```
ai-trading-system-v2/
├── src/
│   ├── core/                    # Motor principal
│   │   ├── market_data_engine.py
│   │   ├── strategy_ensemble.py
│   │   └── execution_engine.py
│   ├── strategies/              # Estrategias de trading
│   │   ├── sma_macd_strategy.py
│   │   ├── ema_strategy.py
│   │   ├── adaptive_sma_strategy.py
│   │   └── bb_rsi_strategy.py
│   ├── filters/                 # Sistema de filtros
│   │   ├── regime_filter.py
│   │   ├── quality_filter.py
│   │   └── temporal_filter.py
│   ├── ai/                      # Modelos de IA
│   │   ├── ensemble_ai.py
│   │   ├── signal_classifier.py
│   │   └── profit_predictor.py
│   ├── risk_management/         # Gestión de riesgo
│   │   ├── position_sizer.py
│   │   └── risk_calculator.py
│   └── dashboard/               # Interface web
│       ├── streamlit_app.py
│       └── components/
├── config/                      # Configuración
│   ├── settings.yaml
│   └── mt5_config.yaml
├── data/                        # Datos históricos
├── models/                      # Modelos entrenados
├── logs/                        # Archivos de log
├── tests/                       # Tests unitarios
└── docs/                        # Documentación
```

## 🛠️ Instalación

### Requisitos Previos
- Python 3.11+
- MetaTrader 5 instalado
- PostgreSQL (opcional, para datos históricos)

### Instalación Rápida
```bash
# Clonar repositorio
git clone https://github.com/santiagoossa-maker/ai-trading-system-v2.git
cd ai-trading-system-v2

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar MT5
cp config/mt5_config.yaml.example config/mt5_config.yaml
# Editar con tus credenciales de MT5

# Ejecutar setup inicial
python setup.py
```

## 🚦 Uso Rápido

### Modo Demo (Sin Riesgo)
```python
from src.core.trading_system import TradingSystem

# Inicializar sistema en modo demo
system = TradingSystem(mode='demo')

# Ejecutar análisis en tiempo real
system.start_analysis()

# Abrir dashboard web
system.launch_dashboard()
```

### Modo Live (Trading Real)
```python
# Solo después de backtesting exitoso
system = TradingSystem(mode='live')
system.start_trading()
```

## 📊 Dashboard Interactivo

El sistema incluye un dashboard web completo:
- **Vista de Mercado 360°**: Estado de todos los activos
- **Monitor de Regímenes**: Identificación automática de condiciones
- **Panel de IA**: Explicaciones de decisiones del modelo
- **Gestor de Riesgo**: Control total del portfolio
- **Análisis de Performance**: Métricas en tiempo real

Acceso: `http://localhost:8501` después de ejecutar `system.launch_dashboard()`

## 🧪 Backtesting Científico

```python
from src.backtesting.scientific_backtester import ScientificBacktester

backtester = ScientificBacktester()

# Walk-forward analysis
results = backtester.walk_forward_analysis(
    start_date='2023-01-01',
    end_date='2024-12-31',
    symbols=['Volatility 75 Index', 'Boom 1000 Index']
)

# Monte Carlo simulation
monte_carlo = backtester.monte_carlo_simulation(
    n_simulations=1000,
    confidence_level=0.95
)

# Análisis de robustez
robustness = backtester.robustness_analysis()
```

## 🔧 Configuración Avanzada

### Configurar Estrategias
```yaml
# config/strategies.yaml
strategies:
  sma_macd:
    enabled: true
    sma_fast: 8
    sma_slow: 50
    macd_fast: 12
    macd_slow: 26
    macd_signal: 9
    
  adaptive_sma:
    enabled: true
    base_period: 20
    volatility_multiplier: 0.5
```

### Configurar IA
```yaml
# config/ai_models.yaml
ensemble_ai:
  signal_classifier:
    model_type: 'xgboost'
    retrain_frequency: '1D'
    features: ['macd_histogram', 'rsi', 'bb_position']
    
  profit_predictor:
    model_type: 'lightgbm'
    target_horizon: 50  # pips
```

## 📈 Resultados Esperados

### Mejoras Inmediatas
- ✅ Reducción de señales falsas: 60-80%
- ✅ Aumento de win rate: 15-25%
- ✅ Drawdowns controlados
- ✅ Adaptabilidad a condiciones cambiantes

### Métricas Objetivo
- **Win Rate**: >70%
- **Profit Factor**: >2.0
- **Sharpe Ratio**: >1.5
- **Max Drawdown**: <15%

## 🗺️ Roadmap de Desarrollo

### Fase 1: Core System (Semana 1-2)
- [x] Estructura base del proyecto
- [ ] Implementar motor de datos MT5
- [ ] Desarrollar estrategias base
- [ ] Sistema de filtros básico
- [ ] Dashboard inicial

### Fase 2: IA Integration (Semana 3-4)
- [ ] Modelos de ensemble AI
- [ ] Sistema de aprendizaje continuo
- [ ] Backtesting científico
- [ ] Optimización de parámetros

### Fase 3: Advanced Features (Semana 5-6)
- [ ] Gestión avanzada de riesgo
- [ ] Múltiples timeframes
- [ ] Alertas inteligentes
- [ ] API para integración externa

### Fase 4: Production Ready (Semana 7-8)
- [ ] Tests comprehensivos
- [ ] Documentación completa
- [ ] Deployment automatizado
- [ ] Monitoreo y alertas

## 🤝 Contribución

Este es un proyecto privado en desarrollo activo. Para contribuir:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📞 Soporte

- **Issues**: Utiliza GitHub Issues para reportar bugs
- **Discusiones**: GitHub Discussions para preguntas generales
- **Email**: Para consultas privadas

## 📄 Licencia

Este proyecto es privado y propietario. Todos los derechos reservados.

---

**⚠️ Disclaimer**: Este sistema es para fines educativos y de investigación. El trading conlleva riesgos significativos. Siempre usa capital que puedas permitirte perder y realiza backtesting exhaustivo antes de trading en vivo.

**🚀 Estado del Proyecto**: En desarrollo activo - Version 2.0.0-alpha