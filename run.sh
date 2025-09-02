for corruption in gaussian_noise motion_blur brightness contrast; do
    for severity in 1 2 3 4 5; do
        echo "Testing $corruption at severity $severity"
        python eval_corrupt.py --config config_corruption.yaml --corruption $corruption --severity $severity
    done
done