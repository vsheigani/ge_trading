# Grammatical Evolution Trading Strategies

This project applies a grammatical evolution (GE) algorithm to generate and backtest trading strategies. 
The implementation of the Grammatical Evolution algorithm is heavily based on PonyGE2, with minor debugging and updates to support recent `pandas` and `numpy` versions. Strategy generation uses a dedicated trading grammar, and backtests are run once per individual to report fitness values separately (a key difference from PonyGE2's multi-objective setup).

The system supports both single-objective and multi-objective optimization. The two objectives used are **CAGR** and **Sharpe Ratio**.

## How It Works

1. `src/ponyge.py` loads parameters and starts the evolutionary search loop.
2. `algorithm/parameters.py` reads the parameters file and wires the selected grammar, operators, and fitness functions.
To find the complete list of Grammatical Evolution parameters, please read the PonyGE2 documentation:
https://github.com/PonyGE/PonyGE2/wiki/Evolutionary-Parameters
3. The grammar in `grammars/trading_grammar.pybnf` generates a Python strategy function. The phenotype is executed to produce:
   - `_exit_bar_`: max holding period in bars (1–9).
   - `_results_`: a long/short signal series (1, -1, 0).
4. The fitness function evaluates the strategy:
   - Single objective: `src/fitness/trading_fitness.py`
   - Multi objective: `src/fitness/multi_objective/moo_trading_fitness.py`
5. Evaluation uses multiprocessing and caching (if enabled) for faster runs.

## Strategy Grammar Details

The grammar produces a `strategy(...)` function and then calls it. It uses:

- **Indicators**: `make_indicator(...)` uses TA-Lib and market fields to build signals. Indicator indices are selected from a list of 155 items, including TA-Lib indicators, OHLCV fields, and time features.
- **Conditions**: `cross_num`, `widen`, `shrink`, `count_positive`, `count_negative`, indicator comparisons, and numeric comparisons.
- **Logic operators**: `&` and `^` to combine conditions.
- **Shifts**: indicators can be shifted by 0–29 bars.
- **Periods and constants**: indicator periods 5–24 and real numbers with two decimal digits.

## Backtest and Fitness Logic

Both fitness functions execute the generated phenotype and backtest it on Apple dollar bars. For multi-objective runs, the backtest happens inside the fitness calculation in `src/fitness/multi_objective/moo_trading_fitness.py`:

- **Dataset**: `datasets/bars/aapl_dollar_bars.h5` (HDF5, key=`key`, indexed by `date_time`, columns: open/high/low/close/volume).  
  The file was generated using the bar sampling repository:  
  https://github.com/vsheigani/ticks_data_sampling_preprocessing
- **Training split**: data before `2016-01-01`.
- **Positioning**: long and short allowed, full-cash position sizing, single active position.
- **Costs**: fixed trade cost `2.0` and slippage `0.0005`.
- **Exit**: positions are closed after `_exit_bar_` bars.
- **Portfolio**: equity curve from cash + marked-to-market positions.

Single-objective (`trading_fitness.py`) evaluates Sharpe Ratio and uses the **minimum Sharpe** over the years `[2009, 2011, 2013]`.  
Multi-objective (`moo_trading_fitness.py`) returns **[Sharpe Ratio, CAGR]** from a single backtest per individual.

## Folder Structure

```
ge/
├── README.md
├── datasets/
│   └── bars/
│       └── aapl_dollar_bars.h5
├── grammars/
│   ├── trading_grammar.pybnf
│   └── ... (other example grammars)
├── parameters/
│   ├── moo/
│   │   └── moo_trading_params.txt
│   └── ... (other PonyGE2-style parameter sets)
├── results/
│   └── run0/ (example outputs: Pareto fronts, PDFs, stats)
├── seeds/
│   └── ... (seed individuals for initial populations)
├── src/
│   ├── ponyge.py
│   ├── algorithm/ (search loop, parameters, and core GE flow)
│   ├── fitness/ (fitness definitions and evaluation pipeline)
│   ├── operators/ (initialisation, selection, crossover, mutation)
│   ├── representation/ (BNF grammar parsing, trees, individuals)
│   ├── scripts/ (utilities for parsing and experiments)
│   ├── stats/ (stats tracking and plotting)
│   └── utilities/
│       ├── trading/ (indicators, metrics, strategy helpers)
│       └── ... (general utilities)
└── ...
```

### Notable files

- `grammars/trading_grammar.pybnf`: Grammar used to generate Python trading strategies.
- `parameters/moo/moo_trading_params.txt`: Multi-objective configuration (NSGA-II selection/replacement, subtree operators, multicore).
- `src/fitness/trading_fitness.py`: Single-objective fitness (min Sharpe across years).
- `src/fitness/multi_objective/moo_trading_fitness.py`: Multi-objective fitness (Sharpe + CAGR) with a single backtest per individual.
- `src/utilities/trading/indicators.py`: TA-Lib indicator construction (`make_indicator`), sometimes referenced as `Create_ta_features`.

## Multi-Objective Configuration (Current Defaults)

`parameters/moo/moo_trading_params.txt` sets:

- Population size: `200`
- Generations: `10`
- Initialisation: ramped half-and-half (`rhh`)
- Crossover/Mutation: subtree-based
- Selection/Replacement: NSGA-II
- Multicore: `True` with `CORES=8`

## Fitness Objectives

- **CAGR** (Compound Annual Growth Rate)
- **Sharpe Ratio**

## Technical Indicators (TA-Lib)

Indicator creation uses TA-Lib via `make_indicator` in `src/utilities/trading/indicators.py`.  
Please refer to the TA-Lib documentation for platform-specific setup instructions.

## Environment Setup (uv)

This project uses `uv` for package management.

1. Install `uv` if you don't already have it:
   - https://docs.astral.sh/uv/
2. Sync dependencies:

```
uv sync
```

## Running the Project

From the repository root:

```
cd src && python ponyge.py --parameters ../parameters/moo/moo_trading_params.txt
```

## Notes

- Fitness evaluation uses `multiprocessing.Pool` when `MULTICORE` is enabled.
- The multi-objective flow performs a single backtest per individual and reports objectives separately (unlike PonyGE2's default multi-objective behavior).

## References

- PonyGE2: https://github.com/PonyGE/PonyGE2
- PonyGE2 documentation: https://github.com/PonyGE/PonyGE2/wiki