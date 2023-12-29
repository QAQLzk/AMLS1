# Import the A task module from A folder
from A.A_py import run_a_task

# Import the B task modules from B folder
from B.B_py import run_b_task
from B.B_py_pretrained import run_b_pretrained_task

def main():
    # Run task A
    print("Running task A...")
    run_a_task()

    # Run task B with the original training process （if want to run, uncomment the following code)
    #print("Running task B with original training process...")
    #run_b_task()

    # Run task B with the pretrained CNN model
    print("Running task B with the pretrained CNN model")
    run_b_pretrained_task()

if __name__ == "__main__":
    main()
