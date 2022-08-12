config=$1

python examples/python/reconstruction_system/run_system.py --config $config --make
python examples/python/reconstruction_system/run_system.py --config $config --register
python examples/python/reconstruction_system/run_system.py --config $config --refine
python examples/python/reconstruction_system/run_system.py --config $config --integrate
python examples/python/reconstruction_system/construct_point_cloud.py --config $config
# python examples/python/reconstruction_system/run_system.py --config $config --slac
# python examples/python/reconstruction_system/run_system.py --config $config --slac_integrate