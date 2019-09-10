#include <stdio.h>
#include <stdlib.h>

//Nesse código cada thread print uma letra na tela. 

__device__ const char *STR = "Hello from GPU";
const char STR_LENGHT = 14;

__device__ void teste(){
	printf("thread x = %d e y = %d\n", threadIdx.x, threadIdx.y);
}


__global__ void cuda_hello(void){
	printf("%c\n", STR[threadIdx.x % STR_LENGHT]);
	teste();
}

int main(void){
	//int num_threads = STR_LENGHT;
	//int num_blocks = 2;
	
//dim3 - tipo de variavel usada para configurar a entrada, ela possui tres parametros(x,y,z)
	dim3 dimBlock(16,16);
	dim3 dimGrid(32,32);
	
//Do lado esquerdo da virgula informamos a configuração do bloco, no lado direito da virgula informamos a configuração das threads. 
	cuda_hello<<<dimGrid,dimBlock>>>(); // Notação para chamada de GPU "<<< >>>()"
	cudaDeviceSynchronize();
	printf("Fim\n");

	return 0;
}
