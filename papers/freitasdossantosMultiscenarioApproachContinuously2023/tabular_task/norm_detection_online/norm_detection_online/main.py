from norm_detection_online.app import DetectionFlow


def main():
    print("Norm Detection Training Procedures Started!!!")
    DetectionFlow.run_first_experiment()
    # DetectionFlow.run_interpretability()

if __name__ == '__main__':
    main()

