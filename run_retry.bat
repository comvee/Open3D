@echo off
set config_path=%1

python examples/python/reconstruction_system/run_system.py --config %config_path% --register
timeout /t 0.5
python examples/python/reconstruction_system/run_system.py --config %config_path% --refine
timeout /t 0.5
python examples/python/reconstruction_system/run_system.py --config %config_path% --integrate