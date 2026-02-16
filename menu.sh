#!/bin/bash

# Menu to run CNN implementations with PyTorch or TensorFlow

# Default dataset path
DEFAULT_DATASET="/home/rocco/Documenti/Dataset/Segmentation"
DATASET_PATH="$DEFAULT_DATASET"

show_dataset_menu() {
    echo ""
    echo "============================================================"
    echo "Dataset Configuration"
    echo "============================================================"
    echo ""
    echo "Current dataset path: $DATASET_PATH"
    echo ""
    echo "  1) Use default path"
    echo "  2) Enter custom dataset path"
    echo "  3) Back to main menu"
    echo ""
    echo "============================================================"
}

show_menu() {
    echo ""
    echo "============================================================"
    echo "CNN Semantic Segmentation - Implementation Menu"
    echo "============================================================"
    echo ""
    echo "Using dataset: $DATASET_PATH"
    echo ""
    echo "Select an implementation to run:"
    echo ""
    echo "  1) PyTorch U-Net Implementation"
    echo "  2) TensorFlow U-Net Implementation"
    echo "  3) Configure Dataset Path"
    echo "  4) Exit"
    echo ""
    echo "============================================================"
}

configure_dataset() {
    while true; do
        show_dataset_menu
        read -p "Enter your choice (1-3): " choice
        
        case $choice in
            1)
                DATASET_PATH="$DEFAULT_DATASET"
                echo ""
                echo "Using default dataset path: $DATASET_PATH"
                read -p "Press Enter to continue..."
                break
                ;;
            2)
                echo ""
                read -p "Enter custom dataset path: " custom_path
                if [ -d "$custom_path" ]; then
                    DATASET_PATH="$custom_path"
                    echo "Dataset path updated to: $DATASET_PATH"
                    read -p "Press Enter to continue..."
                    break
                else
                    echo ""
                    echo "Error: Directory does not exist: $custom_path"
                    read -p "Press Enter to try again..."
                fi
                ;;
            3)
                break
                ;;
            *)
                echo ""
                echo "Invalid choice! Please enter 1, 2, or 3."
                read -p "Press Enter to try again..."
                ;;
        esac
    done
}

run_pytorch() {
    echo ""
    echo "Running PyTorch implementation..."
    echo "Dataset: $DATASET_PATH"
    echo ""
    python3 CNN_pytorch.py "$DATASET_PATH"
}

run_tensorflow() {
    echo ""
    echo "Running TensorFlow implementation..."
    echo "Dataset: $DATASET_PATH"
    echo ""
    python3 CNN_tensorflow.py "$DATASET_PATH"
}

# Main menu loop
while true; do
    clear
    show_menu
    
    read -p "Enter your choice (1-4): " choice
    
    case $choice in
        1)
            run_pytorch
            read -p "Press Enter to return to menu..."
            ;;
        2)
            run_tensorflow
            read -p "Press Enter to return to menu..."
            ;;
        3)
            configure_dataset
            ;;
        4)
            echo ""
            echo "Exiting... Goodbye!"
            exit 0
            ;;
        *)
            echo ""
            echo "Invalid choice! Please enter 1, 2, 3, or 4."
            read -p "Press Enter to try again..."
            ;;
    esac
done
