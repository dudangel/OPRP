#include <stdio.h>
#include <stdlib.h>

__global__ void matrix_mult(float *a, float *b, float *c, float N){
    int index_x = blockIdx.x * blockDim.x + threadIdx.x; //linha
    int index_y = blockIdx.y * blockDim.y + threadIdx.y; //coluna

    float index_z; //index da matriz c (que é a matriz resultado)

    //Verfica se está dentro das dimensões
    if(index_x < N) && (index_y < N){
        //cada thread calcula um elemento do bloco sub-matriz
        for(int i = 0; i < N*N; i++){
            index_z += a[index_x * N + i] * b[i * N + index_y];
        }
    //Coloca o elemento resultado que foi calculado na sua posição na matriz resultado (C)
    c[index_x * N + index_y] = index_z;
    }
    
}

#define N (1024*1024)
#define THREADS_PER_BLOCK 512 //Definimos 512 threads por bloco. Essa informação varia conforme a placa, ese valor é padrão pré definido pelo tipo da placa

int main(){

    float *a, *b, *c; //Estão na memória ram/CPU
    float *d_a, *d_b, *d_c; //Ponteiros para indexar a memória da GPU
    float size = N * N * sizeof(float);

    cudaMalloc((void **) &d_a, size);
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_c, size);

    a = (float *) malloc(size);
    b = (float *) malloc(size);
    c = (float *) malloc(size);

    for(int i = 0; i < N * N; i++){
        a[i] = b[i] = i;
        c[i] = 0;
    }

    //Copia os dados das variaveis da CPU para as variaveis da GPU
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Esq = configurando quantos blocos precisa. Faz-se um arredondamento da conta de quantos blocos são necessários para processar de 512 em 512. 
    //Dir = Configurando quantas threads precisa
    matrix_mult<<< (N + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK ,  THREADS_PER_BLOCK >>>(d_a, d_b, d_c, N);

    //Recupera os dados que foram processador na GPU para a CPU
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    printf( "c[0] = %d\n",c[0] );
	printf( "c[%d] = %f\n",N-1, c[N-1] );

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
