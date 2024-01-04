
def create_training_data():
    for Class in class_names:
        path = os.path.join(tumor, Class)
        class_num = class_names.index(Class)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_Size, IMG_Size))
                training_data.append([new_array, class_num])
            except Exception as e:
                # Print the error and skip the image if loading or resizing fails
                print(f"Error loading/resizing image: {os.path.join(path, img)}")
                continue

create_training_data()

# Shuffle the training data
random.shuffle(training_data)

# Check the number of samples in training_data
print("Number of samples in training_data:", len(training_data))

features_x = []
target_y = []

for features, label in training_data:
    features_x.append(features)
    target_y.append(label)

# Convert lists to NumPy arrays
features_x = np.array(features_x).reshape(-1, IMG_Size, IMG_Size, 1)
target_y = np.array(target_y)

# Save features_x and target_y to pickle files
pickle_out = open('features_x.pickle', 'wb')
pickle.dump(features_x, pickle_out)
pickle_out.close()

pickle_out = open('target_y.pickle', 'wb')
pickle.dump(target_y, pickle_out)
pickle_out.close()


features_x = features_x / 255.0




model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=features_x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(66, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(features_x, target_y, batch_size=100, validation_split=0.1, epochs = 23)

model.save('tumor_detection.h5')





