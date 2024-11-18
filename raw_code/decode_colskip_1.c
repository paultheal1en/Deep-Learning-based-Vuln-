static void decode_colskip(uint8_t* plane, int width, int height, int stride, VC9Context *v){

    int x, y;

    GetBitContext *gb = &v->s.gb;



    for (x=0; x<width; x++){

        if (!get_bits(gb, 1)) //colskip

            for (y=0; y<height; y++)

                plane[y*stride] = 0;

        else

            for (y=0; y<height; y++)

                plane[y*stride] = get_bits(gb, 1);

        plane ++;

    }

}
