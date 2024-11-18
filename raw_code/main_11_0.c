int main(int argc,char* argv[]){

    int i, j;

    uint64_t sse=0;

    uint64_t dev;

    FILE *f[2];

    uint8_t buf[2][SIZE];

    uint64_t psnr;

    int len= argc<4 ? 1 : atoi(argv[3]);

    int64_t max= (1<<(8*len))-1;

    int shift= argc<5 ? 0 : atoi(argv[4]);

    int skip_bytes = argc<6 ? 0 : atoi(argv[5]);



    if(argc<3){

        printf("tiny_psnr <file1> <file2> [<elem size> [<shift> [<skip bytes>]]]\n");

        printf("For WAV files use the following:\n");

        printf("./tiny_psnr file1.wav file2.wav 2 0 44 to skip the header.\n");

        return -1;

    }



    f[0]= fopen(argv[1], "rb");

    f[1]= fopen(argv[2], "rb");

    if(!f[0] || !f[1]){

        fprintf(stderr, "Could not open input files.\n");

        return -1;

    }

    fseek(f[shift<0], shift < 0 ? -shift : shift, SEEK_SET);



    fseek(f[0],skip_bytes,SEEK_CUR);

    fseek(f[1],skip_bytes,SEEK_CUR);



    for(i=0;;){

        if( fread(buf[0], SIZE, 1, f[0]) != 1) break;

        if( fread(buf[1], SIZE, 1, f[1]) != 1) break;



        for(j=0; j<SIZE; i++,j++){

            int64_t a= buf[0][j];

            int64_t b= buf[1][j];

            if(len==2){

                a= (int16_t)(a | (buf[0][++j]<<8));

                b= (int16_t)(b | (buf[1][  j]<<8));

            }

            sse += (a-b) * (a-b);

        }

    }



    if(!i) i=1;

    dev= int_sqrt( ((sse/i)*F*F) + (((sse%i)*F*F) + i/2)/i );

    if(sse)

        psnr= ((2*log16(max<<16) + log16(i) - log16(sse))*284619LL*F + (1<<31)) / (1LL<<32);

    else

        psnr= 100*F-1; //floating point free infinity :)



    printf("stddev:%3d.%02d PSNR:%2d.%02d bytes:%d\n",

        (int)(dev/F), (int)(dev%F),

        (int)(psnr/F), (int)(psnr%F),

        i*len);

    return 0;

}
