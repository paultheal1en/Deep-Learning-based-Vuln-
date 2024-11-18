static void h261_loop_filter_c(uint8_t *src, int stride){

    int x,y,xy,yz;

    int temp[64];



    for(x=0; x<8; x++){

        temp[x      ] = 4*src[x           ];

        temp[x + 7*8] = 4*src[x + 7*stride];

    }

    for(y=1; y<7; y++){

        for(x=0; x<8; x++){

            xy = y * stride + x;

            yz = y * 8 + x;

            temp[yz] = src[xy - stride] + 2*src[xy] + src[xy + stride];

        }

    }



    for(y=0; y<8; y++){

        src[  y*stride] = (temp[  y*8] + 2)>>2;

        src[7+y*stride] = (temp[7+y*8] + 2)>>2;

        for(x=1; x<7; x++){

            xy = y * stride + x;

            yz = y * 8 + x;

            src[xy] = (temp[yz-1] + 2*temp[yz] + temp[yz+1] + 8)>>4;

        }

    }

}
