#include <stdio.h>
#include <stdlib.h>
//Otimizando o uso de memória

__global__ void matrix_mult(float *a, float *b, float *c, float N){
    //O uso do shared caracteriza que a variavel (no caso essa submatriz/sub bloco é criado na meória compartilhada que a GPU tem dentro dela)
    __shared__ float sub_a[N_sub][N_sub];
    __shared__ float sub_b[N_sub][N_sub];

    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    //Identifica a linha e coluna do elemento da matriz resposta (C) que ele vai calcular
    int index_x = block_y * N_sub + thread_y; //linha
    int index_y = block_x * N_sub + thread_x; //coluna

    float index_z = 0;
    //Loop na matriz a e na B pra calcular o elemento resultado que vai na C
    for(int sub_block = 0; sub_block < N/N_sub; sub_block++){
        //Calculando o valor do elemento do sub bloco(sub matriz) que são buscados na matriz a e na matriz b
        //Isso é feito dentro da memória compartilhada
        sub_a[thread_y][thread_x] = a[index_x * N + sub_block * N_sub + thread_x];
        sub_b[thread_y][thread_x] = b[(sub_block * N_sub + thread_y) * N + index_x];
        __syncthreads();

        for(int i = 0;  i < N_sub; i++){
            index_z += sub_a[thread_y][i] + sub_b[i][thread_x];
        }
        __syncthreads();
    }

    //Coloca o elemento resultado que foi calculado na sua posição na matriz resultado (C)
    c[index_x * N + index_y] = index_z;

}

#define N_sub 32
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
