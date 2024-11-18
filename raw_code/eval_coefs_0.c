static void eval_coefs(int *coefs, const int *refl)

{

    int buffer[10];

    int *b1 = buffer;

    int *b2 = coefs;

    int x, y;



    for (x=0; x < 10; x++) {

        b1[x] = refl[x] << 4;



        for (y=0; y < x; y++)

            b1[y] = ((refl[x] * b2[x-y-1]) >> 12) + b2[y];



        FFSWAP(int *, b1, b2);

    }



    for (x=0; x < 10; x++)

        coefs[x] >>= 4;

}
