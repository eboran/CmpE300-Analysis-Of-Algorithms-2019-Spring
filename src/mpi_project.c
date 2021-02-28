/*
Student Name: Emre Boran
Student Number: 2013300075
Compile Status: Compiling
Program Status: Working
*/

#include <mpi.h>
#include <stdio.h>


#include <stdlib.h>
#include <string.h>
#include <errno.h>


const int MAX_LINE_LEN = 6000;
const int MAX_WORD_LEN = 20;
const int NUM_WORDS = 1000;
const int EMBEDDING_DIMENSION= 300;
const char DELIMITER[2] = "\t";

const int COMMAND_EXIT= 0;
const int COMMAND_QUERY= 1;
const int COMMAND_CALCULATE_SIMILARITY= 2;


void distributeEmbeddings(char *filename,int world_size){
	int partLen = NUM_WORDS/(world_size-1);
	char line[MAX_LINE_LEN];
	//Reading embedding file
	
	FILE *file = fopen (filename, "r" );
	int wordIndex = 0;
    //Sending the necessary data to process
    float* embeddings_matrix = (float*)malloc(sizeof(float) * NUM_WORDS*EMBEDDING_DIMENSION);

    char* words = (char*)malloc(sizeof(char) * NUM_WORDS*MAX_WORD_LEN);

    
    for(int i = 0; i<NUM_WORDS; i++){
	    fgets(line, MAX_LINE_LEN, file);

      	char *word;
      	word = strtok(line, DELIMITER);

      	strcpy(words+i*MAX_WORD_LEN, word);

      
      	for(int embIndex = 0; embIndex<EMBEDDING_DIMENSION; embIndex++){
	    	char *field = strtok(NULL, DELIMITER);
	       	float emb = strtof(field,NULL);
	       	*(embeddings_matrix+i*EMBEDDING_DIMENSION+embIndex) = emb;
      	}
	}
    for(int p = 1;p<=world_size-1;p++){
	    
	    char *subwords;
	    subwords = (char*)malloc(sizeof(char) * partLen * MAX_WORD_LEN);
      	
      	for (int i = 0;i<partLen*MAX_WORD_LEN;i++){
      		*(subwords+i) = *(words+(p-1)*MAX_WORD_LEN*partLen+i);
      	}
        
	    //Sending words to process...
	    MPI_Send(
	         	/* data         = */ subwords, 
	      		/* count        = */ partLen*MAX_WORD_LEN, 
	      		/* datatype     = */ MPI_CHAR,
	      		/* destination  = */ p, 
	      		/* tag          = */ 0, 
	      		/* communicator = */ MPI_COMM_WORLD);

	    //Sending embeddings to process


        float* subarray = (float*)malloc(sizeof(float) * partLen*EMBEDDING_DIMENSION);

	    for(int m = 0; m< partLen*EMBEDDING_DIMENSION ;m++){
	    	*(subarray+m) = *(embeddings_matrix + (p-1)*partLen*EMBEDDING_DIMENSION+m) ;
	    }

	    MPI_Send(
	         	/* data         = */ subarray, 
	      		/* count        = */ partLen * EMBEDDING_DIMENSION, 
	      		/* datatype     = */ MPI_FLOAT,
	      		/* destination  = */ p,
	      		/* tag          = */ 0,
	      		/* communicator = */ MPI_COMM_WORLD);
   	}
   //Embedding file.. has been distributed
}




int findWordIndex(char *words, char *query_word,int partLen){
	    for(int wordIndex = 0; wordIndex<partLen; wordIndex++){
	        if(strcmp((words+wordIndex*MAX_WORD_LEN), query_word)==0){
	        return wordIndex;
	    }
	}
    return -1;
}


int runMasterNode(int world_rank,int world_size){

    // If we are rank 0, set the number to -1 and send it to process 1
	int partLen = NUM_WORDS / world_size-1;
	int slaveSize = world_size-1;
    distributeEmbeddings("./word_embeddings_1000.txt",world_size);

    while(1==1){
        printf("Please type a query word:\n");
	    
        char queryWord[256];
        scanf( "%s" , queryWord);
        printf("Query word:%s\n",queryWord);
        char exit1[256] = "EXIT";
        char exit2[256] = "exit";
        
        if(strcmp(exit1,queryWord)==0 || strcmp(exit2,queryWord)==0){ //EXİT gönder 

        	for (int p = 1;p<=world_size-1;p++){
		        MPI_Send(
		          /* data         = */ (void*)&COMMAND_EXIT, 
		          /* count        = */ 1, 
		          /* datatype     = */ MPI_INT, 
		          /* destination  = */ p,
		          /* tag          = */ 0, 
		          /* communicator = */ MPI_COMM_WORLD);
        	}
        	return 0;
        }else{

	        for(int p = 1; p<=slaveSize;p++){
		        //Command is being sent to process 
		        MPI_Send( (void *)&COMMAND_QUERY,  1, MPI_INT, p,0, MPI_COMM_WORLD);
		        MPI_Send( queryWord,MAX_WORD_LEN,MPI_CHAR,p,0,MPI_COMM_WORLD);
		        //Query is sent to process
			}

			float *query_embeddings =  (float*)malloc(sizeof(float)*EMBEDDING_DIMENSION);
			
			int found =0;

			for (int p = 1;p<=slaveSize;p++){
				int *target_word_index =  (int*)malloc(sizeof(int));
		        MPI_Recv(target_word_index, 1,MPI_FLOAT,p,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		                  
				if (*target_word_index >= 0){
					MPI_Recv(query_embeddings,EMBEDDING_DIMENSION,MPI_FLOAT,p,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			                  
					//query command is received 
					
					found = 1;
				}
			}

			if (!found){
				printf("Query word was not found\n");
				continue;
			}
			for (int p=1;p<=slaveSize;p++){
			    MPI_Send((void *)&COMMAND_CALCULATE_SIMILARITY,1,MPI_INT,p,0,MPI_COMM_WORLD);
		                 

		        //COMMAND_CALCULATE_SIMILARITY is sent to process 
		              
				MPI_Send(query_embeddings,EMBEDDING_DIMENSION,MPI_FLOAT,p,0,MPI_COMM_WORLD);
						         	 
			}
			
	        float *bestPscore = (float*)malloc(sizeof(float) * (world_size-1)*(world_size-1));
			char* words = (char*)malloc(sizeof(char) * MAX_WORD_LEN*(world_size-1)*(world_size-1));

			for(int p=1;p<=slaveSize;p++){

				MPI_Recv(bestPscore+(p-1)*slaveSize, slaveSize, MPI_FLOAT,p,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);	
				MPI_Recv(words+(p-1)*slaveSize*MAX_WORD_LEN, MAX_WORD_LEN*slaveSize, MPI_CHAR,p,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				//Best P Scores and Words are received
			}

			//ALL subarrays are received BY MASTER NODE

			int ptrScore[world_size-1]; //pointer array for maximum of subarrays received by slaves

	        float *outputPscore = (float*)malloc(sizeof(float) * slaveSize);
			char *outputWords = (char*)malloc(sizeof(char) * MAX_WORD_LEN*slaveSize);

			//initilization ptrScore is 0-10-20 ... 90. It holds the starting point of subarrays. 
			for(int i = 0;i<slaveSize;i++){
				ptrScore[i] = i*slaveSize;
			}
			
			//starting of finding part of most relevant p words 
			int maxIndex;//it is the index of next maximum relevant word
			for(int j =0;j<slaveSize;j++){
				float maxSimilarityScore = -1;

				for(int i = 0;i<slaveSize;i++){
					if(*(bestPscore + ptrScore[i])>maxSimilarityScore){
						maxSimilarityScore = *(bestPscore + ptrScore[i]);
						maxIndex = i;
					}
				}
				*(outputPscore + j) = *(bestPscore + ptrScore[maxIndex]);
				
				for(int i = 0;i<MAX_WORD_LEN;i++){
					*(outputWords + j*MAX_WORD_LEN + i) = *(words + ptrScore[maxIndex] * MAX_WORD_LEN+i);
				}
				
				ptrScore[maxIndex]++;
			}
			//end of finding most relevant words

			//printing result
			printf("TOP %d RESULTS:\n",slaveSize);
			for(int i= 0;i<world_size-1;i++){
			    int c = i * MAX_WORD_LEN;
			    while (*(outputWords+c) != '\0') {
		  		  	printf("%c", outputWords[c]);
		    		c++;
	   			}		    	
				printf(": %f\n", *(outputPscore+i));
			}
		}

   }
   return 0;
}

int runSlaveNode(int world_rank,int world_size){

	int partLen = NUM_WORDS / (world_size-1);  
	int slaveSize = world_size-1;
	//Receiving words  

	char* words = (char*)malloc(sizeof(char) * MAX_WORD_LEN*partLen);
  
    MPI_Recv(words, partLen*MAX_WORD_LEN, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	//Words received 

	float* embeddings_matrix = (float*)malloc(sizeof(float) * partLen*EMBEDDING_DIMENSION);

	//Process started to receive embedding part
	  
	MPI_Recv(embeddings_matrix, partLen*EMBEDDING_DIMENSION, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

	//Process received embedding part


	while(1==1){

	    //Process is waiting for command

	    int command;

	    MPI_Recv(
	          /* data         = */ &command, 
	          /* count        = */ 1, 
	          /* datatype     = */ MPI_INT, 
	          /* source       = */ 0, 
	          /* tag          = */ 0, 
	          /* communicator = */ MPI_COMM_WORLD, 
	          /* status       = */ MPI_STATUS_IGNORE);
	    
	    //Command received

	    if(command == COMMAND_EXIT){
	      	return 0;
	    }else if(command == COMMAND_QUERY){
	    	//printf("command=1 \n");
	      	char* query_word = (char*)malloc(sizeof(char) * MAX_WORD_LEN);

	        MPI_Recv(
	          /* data         = */ query_word, 
	          /* count        = */ MAX_WORD_LEN, 
	          /* datatype     = */ MPI_CHAR, 
	          /* source       = */ 0, 
	          /* tag          = */ 0, 
	          /* communicator = */ MPI_COMM_WORLD, 
	          /* status       = */ MPI_STATUS_IGNORE);

	       	//Query word received

	       	int wordIndex = findWordIndex(words, query_word,partLen);


	        float* query_embeddings = (embeddings_matrix+wordIndex*EMBEDDING_DIMENSION);
	        	
		    int* WordIndexInMasterNode = (int*)malloc(sizeof(int));
		    *WordIndexInMasterNode = wordIndex*world_rank;

		    MPI_Send(WordIndexInMasterNode, 1,MPI_INT,0,0,MPI_COMM_WORLD);
		         	
		    if(*WordIndexInMasterNode >=0 ){			
				MPI_Send(query_embeddings, EMBEDDING_DIMENSION, MPI_FLOAT,0,0,MPI_COMM_WORLD);			
		    }
		    
	    }else if(command == COMMAND_CALCULATE_SIMILARITY){

			
			float* query_embeddings = (float*)malloc(sizeof(float) *EMBEDDING_DIMENSION);

			MPI_Recv(query_embeddings,EMBEDDING_DIMENSION,MPI_FLOAT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	            
		
	        //Calculating similarities by using a way similar to insertion sort. It finds most relevant P words among partLen words.
	        
	        int mostSimilarWordIndex = -1;
	        float maxSimilarityScore = -1;

	        float* similarityScores = (float*)malloc(sizeof(float) * partLen);

	        int *bestPindex = (int*)malloc(sizeof(int) * world_size-1); //best P words index 
	        float *bestPscore = (float*)malloc(sizeof(float) * world_size-1);//best P words scores


	        //initilization

	        for(int i = 0;i<world_size-1;i++){
	        	*(bestPscore+i) = -1;
	        	*(bestPindex+i) = -1;
	        }


	        int last=-1;//it represent whether the array which has length P is full or not

	        for(int wordIndex = 0; wordIndex<partLen; wordIndex++){
	            float similarity = 0.0;

	            for(int embIndex = 0; embIndex<EMBEDDING_DIMENSION; embIndex++){
	            	float emb1 = *(query_embeddings + embIndex);
	            	float emb2 = *(embeddings_matrix + wordIndex*EMBEDDING_DIMENSION + embIndex);
	            	similarity +=(emb1*emb2);
	            }

	            *(similarityScores + wordIndex) = similarity;


	            if(similarity>maxSimilarityScore){
	            	mostSimilarWordIndex = wordIndex;
	            	maxSimilarityScore=similarity;
	            }
	            int position;


	            // if bestPscore and bestPindex is full and new element should be inside of array, 
	            // we need to be careful the last element of bestPscore and bestPindex when shifting arrays

	            if (last < world_size-2){ // if array is not full.. 
	            	position = last;
	            	last++;
	            
					while ( position >=0 && similarity > *(bestPscore+position) ){
	        			*(bestPscore + position + 1) = *(bestPscore + position);
	        			*(bestPindex + position + 1) = *(bestPindex + position);
	            		position--;
            		}
	         
	            }else{ // if the array is full, search for a right place to insert by starting last. It is like insertion sort.
            		position = world_size-2;
        		
            		while ( position >=0 && similarity > *(bestPscore+position) ){
		        		if(position != world_size-2){ // carefuling for last element of arrays
		        			*(bestPscore + position + 1) = *(bestPscore + position);
		        			*(bestPindex + position + 1) = *(bestPindex + position);
		            	}
	            		position--;
            		}
	         		
            	}
				
				// if position == world_size-2, then new word's score is less than all words' scores in the bestPscore

            	if(position != world_size-2){//insert new words and scores to the appropriate position which is position + 1, it is the same logic in insertion sort. 
	            	*(bestPscore + position+1) = similarity;
	        		*(bestPindex + position+1) = wordIndex;	
            	}
            		            
	        }

	        
        	char* sortedwords = (char*)malloc(sizeof(char) * MAX_WORD_LEN*world_size-1);

	    	for(int i=0;i<world_size-1;i++){
	    		int nextIndex = *(bestPindex+i);
	    	 	for(int j = 0; j< MAX_WORD_LEN; j++){
	    	 	 	*(sortedwords+i*MAX_WORD_LEN +j) = *(words + (nextIndex * MAX_WORD_LEN) +j);
	    	 	 }
	    	}

	    	//send the bestPscore and sortedwords to master node

			MPI_Send(bestPscore, world_size-1, MPI_FLOAT,0,0,MPI_COMM_WORLD);
			MPI_Send(sortedwords, (world_size-1) * MAX_WORD_LEN, MPI_CHAR,0,0,MPI_COMM_WORLD);
	      	
	    }


	}

	return 0 ;
}



int main(int argc, char** argv) {


    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message

    // We are assuming at least 2 processes for this task
    if (world_size < 2) {
        fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int wordIndex;

    if (world_rank == 0) {

    	runMasterNode(world_rank,world_size);

    }else {

	    runSlaveNode(world_rank,world_size);

	}
      
    MPI_Finalize();

    //Processors stopped
}


