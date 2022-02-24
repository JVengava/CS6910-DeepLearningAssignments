if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    (train_Input, train_Output), (test_Input, test_Output) = fashion_mnist.load_data()

    # visualisation of the classes:
    img_classes_fig = plt.figure("Sample image per class")
    grid_specification = img_classes_fig.add_gridspec(2, 5)

    total_num_classes = int(np.max(train_Output) + 1)
    
    class_dictionary = {
        "0": "T-shirt/Top",
        "1": "Trouser",
        "2": "Pullover",
        "3": "Dress",
        "4": "Coat",
        "5": "Sandal",
        "6": "Shirt",
        "7": "Sneaker",
        "8": "Bag",
        "9": "Ankle Boot",
    }   

    Image_array = []
    Image_class_label_array = []
    for i in range(total_num_classes): # iteration over categories -- pick a category
      if i < 5:   # i variable is mainly used to reserve the slot in the image plot for ith category
        globals()["ax" + str(i + 1)] = img_classes_fig.add_subplot(grid_specification[0, i])
      elif i >= 5:
        globals()["ax" + str(i + 1)] = img_classes_fig.add_subplot(grid_specification[1, i - 5])
      
      for j in range(train_Input.shape[0]): # iteration over 60000 images of size (28X28)
          if train_Output[j] == i:
            plt.imshow(train_Input[j], cmap="gray")
            plt.xlabel(class_dictionary[str(i)], fontsize=15)
            Image_array.append(train_Input[j])
            Image_class_label_array.append(class_dictionary[str(i)])
            break

    plt.show()

    wandb.init(project="CS6910_DeepLearning_Assignment1", entity="cs21z032_cs22z005")
    wandb.log({"Sample image per class":[wandb.Image(img,caption = caption) for img,caption in zip(Image_array, Image_class_label_array)]})
