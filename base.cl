kernel void mulmat( global const int *P, global const int *Q, global int *R, int N){
    int id = get_global_id(0);
    int tmp = 0;
    for(int i = 0; i< N ;i++){
        tmp += P[id%N + i * N] * Q[id - id%N + i];
    }
    R[id] = tmp;
}

//P[id%N + i * N]
// id%N allows to start on the right line 
// i*N is the iteration through the line

//Q[id - id%N + i]
//id - id%N allows to go back to the top of the column
// +i is the iteration through the column

