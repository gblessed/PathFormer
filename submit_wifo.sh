#!/bin/bash
#SBATCH --job-name=wifo-gen-all
#SBATCH --output=/home/blessedg/Pathformer/logs/wifo_beam_%A_%a.out
#SBATCH --error=/home/blessedg/Pathformer/logs/wifo_beam_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:nvidia:1

source ~/.bashrc
conda activate pathformer
set -euo pipefail


cd /home/blessedg/Pathformer
# python generate_wifo_data.py

SCENARIOS=(
    'city_0_newyork_3p5_s'
    'city_1_losangeles_3p5'
    'city_2_chicago_3p5'
    'city_3_houston_3p5'
    'city_4_phoenix_3p5'
    'city_5_philadelphia_3p5'
    'city_6_miami_3p5'
    'city_7_sandiego_3p5'
    'city_8_dallas_3p5'
    'city_9_sanfrancisco_3p5'
    'city_10_austin_3p5'
    'city_11_santaclara_3p5'
    'city_12_fortworth_3p5'
    'city_13_columbus_3p5'
    'city_17_seattle_3p5_s'
    'city_18_denver_3p5'
    'city_19_oklahoma_3p5_s'
    'city_16_sanfrancisco_3p5_lwm'
    'city_23_beijing_3p5'
    'city_31_barcelona_3p5'
    'city_35_san_francisco_3p5'
    'city_47_chicago_3p5'
    'city_89_nairobi_3p5'
    'city_91_xiangyang_3p5'
    'city_92_sãopaulo_3p5'
    'boston5g_3p5'
    'city_86_ankara_3p5'
    'city_72_capetown_3p5'
    'city_84_baoding_3p5'
    'city_95_delhi_3p5'
    'city_96_osaka_3p5'
    'city_88_tongshan_3p5'
)
TEST_SCENARIOS=(
    'city_1_losangeles_3p5'
    'city_2_chicago_3p5'
    'city_3_houston_3p5'
    'city_4_phoenix_3p5'
    'city_5_philadelphia_3p5'
    'city_6_miami_3p5'
    'city_7_sandiego_3p5'
    'city_8_dallas_3p5'
    'city_9_sanfrancisco_3p5'
    'city_10_austin_3p5'
    'city_11_santaclara_3p5'
    'city_12_fortworth_3p5'
    'city_13_columbus_3p5'
    'city_18_denver_3p5'
)
for SCENARIO in "${TEST_SCENARIOS[@]}$"; do
    echo "Generating data. for scenario $SCENARIO"
    python generate_wifo_sub6_to_mmwave_data.py --scenario $SCENARIO
    done