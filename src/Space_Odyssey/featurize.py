"""
This file contains the code different image featurization methods.

A) Graph Based Features
B) Topological Features
C) ScimCLR Embedding
D) UNI Model Embedding

"""

#TODO:
# adapt whole thing to batch processing!!!!!!!!!!!!!!!!
#add spectral features



class featurizer:
    def __init__(self, config):
        self.config = config



    #A)GRAPH Based Features########################################################################################################################
    def Img2graph(self, image,graph_type):
      
        class WeisfeilerLehmanMachine:
            """
            Weisfeiler Lehman feature extractor class.
            """
            def __init__(self, graph, features, iterations):
                """
                Initialization method which also executes feature extraction.
                :param graph: The Nx graph object.
                :param features: Feature hash table.
                :param iterations: Number of WL iterations.
                """
                self.iterations = iterations
                self.graph = graph
                self.features = features
                self.nodes = self.graph.nodes()
                self.extracted_features = [str(v) for k, v in features.items()]
                self.do_recursions()

            def do_a_recursion(self):
                """
                single WL recursion
                :return new_features: The hash table with extracted WL features.
                """
                new_features = {}
                for node in self.nodes:
                    neighbors = self.graph.neighbors(node)
                    degs = [self.features[neb] for neb in neighbors]
                    
                    features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
                    features = "_".join(features)
                    hash_object = hashlib.md5(features.encode())
                    hashing = hash_object.hexdigest()
                    new_features[node] = hashing
                self.extracted_features = self.extracted_features + list(new_features.values())
                return new_features

            def do_recursions(self):
                """
                series of WL recursions
                """
                for _ in range(self.iterations):
                    self.features = self.do_a_recursion()
  
    
        def rag_boundary_graph(image, segments):
            """Create Region Adjacency Graph using boundary information"""
            # Handle channel ordering
            if image.ndim == 3 and image.shape[0] in [3, 4]:  # Channel-first format
                image = np.moveaxis(image, 0, -1)  # Convert to channel-last format
            
            # Convert to grayscale for edge detection
            #FIXIT: get the edges from the segmentation !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            gray_img = color.rgb2gray(image) if image.ndim == 3 else image
            edge_map = filters.sobel(gray_img)
            
            # Create RAG
            rag = skgraph.rag_boundary(segments, edge_map)
            
            # Create adjacency matrix with proper indexing
            n_segments = len(np.unique(segments))
            adj_matrix = np.zeros((n_segments, n_segments))
            
            for region1, region2, weight in rag.edges(data='weight'):
                # Convert from 1-based to 0-based indexing
                r1 = region1 - 1
                r2 = region2 - 1
                if r1 < n_segments and r2 < n_segments:
                    adj_matrix[r1, r2] = 1
                    adj_matrix[r2, r1] = 1
            
            return adj_matrix, rag
        

        def superpixel_graph(segments):
            # Find k=5 nearest neighbors for each centroid
            k = 5
            # Fix: Get actual number of superpixels
            unique_segments = np.unique(segments)
            n_segments = len(unique_segments)
            print("n_segments: ", n_segments)
            
            # Get centroids of superpixels
            superpixels = [np.argwhere(segments == i) for i in unique_segments]
            centroids = np.array([np.mean(sp, axis=0) for sp in superpixels])
            
            # Compute pairwise distances between centroids
            #TODO:Replace with own GPU implementation .... wherever that file is :/
            dist_matrix = pairwise_distances(centroids)
            
            
            # Initialize adjacency matrix
            graph = np.zeros((n_segments, n_segments))
        


            
            # For each centroid, connect to k nearest neighbors
            for i in range(n_segments):
                # Get indices of k nearest neighbors (excluding self)
                print("i: ",i)
                print(dist_matrix.shape)
                nearest_neighbors = np.argsort(dist_matrix[i])[1:k+1]
                
                # Add edges to graph (make it symmetric)
                for j in nearest_neighbors:
                    if j < n_segments:
                        graph[i,j] = graph[j,i] = 1
        
            return graph,dist_matrix   

    def Graph2Vec(self, graph,vector_dimension,):

        #WL pulled ou to Img2graph adjust for this

        """
        Uses graph2vec https://github.com/benedekrozemberczki/graph2vec
        Weilsfeiler Lehman Machine interative node relabeling based on neighbor node features
        and the node itself.
        Node feature are strings that in each iteration are updated with concatenation of
        the node itself + the neighbor node features, followed by MD5 hashing.


        -> Output vector generated by Doc2Vec on the list of node features = document
        """
    


    def spectral_features(self, graph):
        pass


    def spectral_features2vec(self, spectral_features):
        pass






    #B) TOPOLOGICAL FEATURES########################################################################################################################
    def PD_2Vec(self, image,filtration_type):
        #Binarize IMAGE
            def binarize_images(X):
                binarized_images=[]
                for img in X:
                    gray_img = np.sum(img, axis=1)
                    gray_img = (gray_img / gray_img.max()) * 255
                    gray_img = gray_img.astype(np.uint8)
                    binarizer = Binarizer(threshold=0.6)
                    binary_img = binarizer.fit_transform(gray_img)
                    binarized_images.append(binary_img)
                return np.array(binarized_images)
      
            # Function to compute persistence diagrams and Shannon entropies
            def compute_filtrations_and_entropies(binary_images,filtration_type):
                """
                Compute the persistence diagrams and Shannon entropies for the binary images.
                Parameters:
                    binary_images: list of binary images
                    filtration_type: type of filtration to apply
                Returns:
                    radial_entropies: list of radial entropies
                    height_entropies: list of height entropies
                    erosion_entropies: list of erosion entropies
                """
            
                # Initialize filtrations
                radial_filtration = RadialFiltration()
                height_filtration = HeightFiltration()
                erosion_filtration = ErosionFiltration()

                # Apply filtrations
                radial_images = radial_filtration.fit_transform(binary_images)
                height_images = height_filtration.fit_transform(binary_images)
                erosion_images = erosion_filtration.fit_transform(binary_images)

                # Initialize persistence homology transformer
                persistence = VietorisRipsPersistence()

                # Compute persistence diagrams
                radial_diagrams = persistence.fit_transform(radial_images)
                height_diagrams = persistence.fit_transform(height_images)
                erosion_diagrams = persistence.fit_transform(erosion_images)

                # Initialize persistence entropy transformer
                entropy = PersistenceEntropy()

                # Compute Shannon entropies
                radial_entropies = entropy.fit_transform(radial_diagrams)  
                height_entropies = entropy.fit_transform(height_diagrams)
                erosion_entropies = entropy.fit_transform(erosion_diagrams)

                return np.array(radial_entropies), np.array(height_entropies), np.array(erosion_entropies)


        #TODO: extend by other summary functions of Persistnece Diagrams (PDs)








    #C) SCIMCLR EMBEDDING########################################################################################################################
    def simclr_embedding(self, image,path_to_model,pca_out=True):
        """
        Parameters:
            image: image to embed
            path_to_model: path to the model
            pca_out: if True, perform PCA on the embeddings
        Returns:
            embeddings: embeddings of the image
        """

        def load_model_weights(model, weights):

            model_dict = model.state_dict()
            weights = {k: v for k, v in weights.items() if k in model_dict}
            if weights == {}:
                print('No weight could be loaded..')
            model_dict.update(weights)
            model.load_state_dict(model_dict)

            return model


        def perform_embedding_extraction(model, train_loader, n_components=2):
            # Collect all images and features
            all_images = []
            all_features = []
            all_labels = []
            
            model.eval()
            with torch.no_grad():
                for images, labels in train_loader:
                    images = images.cuda()
                    # Store original images
                    all_images.append(images.cpu().numpy())
                    all_labels.append(labels.numpy())
                    
                    # Get features from model
                    if RETURN_PREACTIVATION:
                        features = model(images)
                    else:
                        # Get features from the layer before the final classification layer
                        features = model.conv1(images)
                        features = model.bn1(features)
                        features = model.relu(features)
                        features = model.maxpool(features)
                        features = model.layer1(features)
                        features = model.layer2(features)
                        features = model.layer3(features)
                        features = model.layer4(features)
                        features = model.avgpool(features)
                        features = torch.flatten(features, 1)
                    
                    all_features.append(features.cpu().numpy())

            all_features = np.concatenate(all_features, axis=0)
            all_images = np.concatenate(all_images, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            return all_features, all_images, all_labels
            
        #PCA Processing ###########################################################
        def perform_pca_analysis(all_features, n_components=2):
            # Apply PCA to reduce dimensions
            print("Applying PCA to SimCLR embeddings...")
            scaler = StandardScaler()
            all_features_scaled = scaler.fit_transform(all_features)
            pca_model = PCA(n_components=n_components, random_state=42)
            all_features_pca = pca_model.fit_transform(all_features_scaled)

            return all_features_pca

    


        def get_pretrained_simclr_embeddings(X,y,MODEL_PATH,RETURN_PREACTIVATION=True):
            model = torchvision.models.__dict__['resnet18'](weights=None)

            state = torch.load(MODEL_PATH, weights_only=False)

            state_dict = state['state_dict']
            for key in list(state_dict.keys()):
                state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

            model = load_model_weights(model, state_dict)

            if RETURN_PREACTIVATION:
                model.fc = torch.nn.Sequential()

            model = model.cuda()
            # Create dataset with both images and labels
            dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
            train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

            all_features, all_images, all_labels = perform_embedding_extraction(model, train_loader)    

            all_features_pca = perform_pca_analysis(all_features)

            return all_features_pca


        
        embeddings = get_pretrained_simclr_embeddings(X,y,path_to_model)

        np.savetxt(f'{out_dir}/simclr_pretrained_pca_embeddings.csv', embeddings, delimiter=',')
        print(f"SIMCLR pretrained PCA embeddings saved to simclr_pretrained_pca_embeddings.csv")



    #D) UNI Model Embedding########################################################################################################################
    def UNI_model_embedding(self, image,path_to_model):
        """
        Parameters:
            image: image to embed
            path_to_model: path to the model
        Returns:
            embeddings: embeddings of the image
        """
        pass




