static int xan_huffman_decode(unsigned char *dest, unsigned char *src)

{

    unsigned char byte = *src++;

    unsigned char ival = byte + 0x16;

    unsigned char * ptr = src + byte*2;

    unsigned char val = ival;

    int counter = 0;



    unsigned char bits = *ptr++;



    while ( val != 0x16 ) {

        if ( (1 << counter) & bits )

            val = src[byte + val - 0x17];

        else

            val = src[val - 0x17];



        if ( val < 0x16 ) {

            *dest++ = val;

            val = ival;

        }



        if (counter++ == 7) {

            counter = 0;

            bits = *ptr++;

        }

    }



    return 0;

}
