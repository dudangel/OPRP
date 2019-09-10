#include <stdio.h>
#include <stdlib.h>

//Memória não compartilhada pois não pode-se usar um ponteiro da gpu para cpu e vice versa.
//Memória distribuida, pois o tempo de acesso as memórias é completamente diferente

//para se encontrar, ela calcula a posição de index no bloco e posteriormente no vetor de threads.
__global__ void vector_add(int *a, int *b, int *c){
    /*Na dimensão x, qual é o numero do bloco? Qual a dimensão/tamanho do bloco? Qual é o numero da thread, ou seja, Qual thread eu sou? 
    Esse é o paralelismo de uma instrução sendo executada por todo mundo (SIMD). 
    Com essas informações ele consegue se localizar dentro da arquitetura da GPU*/
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    c[index] = a[index] + b[index];

}

#define N (2048*2048)
#define THREADS_PER_BLOCK 512 //Definimos 512 threads por bloco. Essa informação varia conforme a placa, ese valor é padrão pré definido pelo tipo da placa

int main(){

    int *a, *b, *c; //Estão na memória ram/CPU
    int *d_a, *d_b, *d_c; //Ponteiros para indexar a memória da GPU
    int size = N * sizeof(int);

    //Usa-se cudaMalloc pois a memória é outra, dentro da GPU. Se fosse na cpu usariamos o malloc normal. 
    cudaMalloc((void **) &d_a, size);
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_c, size);

    a = (int *) malloc(size);
    b = (int *) malloc(size);
    c = (int *) malloc(size);

    for(int i = 0; i < N; i++){
        a[i] = b[i] = i;
        c[i] = 0;
    }

    //Copia os dados das variaveis da CPU para as variaveis da GPU
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Esq = configurando quantos blocos precisa. Faz-se um arredondamento da conta de quantos blocos são necessários para processar de 512 em 512. 
    //Dir = Configurando quantas threads precisa
    vector_add<<< (N + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK ,  THREADS_PER_BLOCK >>>(d_a, d_b, d_c);

    //Recupera os dados que foram processador na GPU para a CPU
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    printf( "c[0] = %d\n",c[0] );
	printf( "c[%d] = %d\n",N-1, c[N-1] );

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
