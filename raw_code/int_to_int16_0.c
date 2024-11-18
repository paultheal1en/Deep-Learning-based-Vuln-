static void int_to_int16(int16_t *out, const int *inp)

{

    int i;



    for (i=0; i<30; i++)

        *(out++) = *(inp++);

}
