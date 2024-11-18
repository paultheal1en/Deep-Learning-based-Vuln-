static int sad16_altivec(void *v, uint8_t *pix1, uint8_t *pix2, int line_size, int h)

{

    int i;

    int s;

    const vector unsigned int zero = (const vector unsigned int)vec_splat_u32(0);

    vector unsigned char perm1, perm2, *pix1v, *pix2v;

    vector unsigned char t1, t2, t3,t4, t5;

    vector unsigned int sad;

    vector signed int sumdiffs;



    sad = (vector unsigned int)vec_splat_u32(0);





    for (i = 0; i < h; i++) {

        /* Read potentially unaligned pixels into t1 and t2 */

        perm1 = vec_lvsl(0, pix1);

        pix1v = (vector unsigned char *) pix1;

        perm2 = vec_lvsl(0, pix2);

        pix2v = (vector unsigned char *) pix2;

        t1 = vec_perm(pix1v[0], pix1v[1], perm1);

        t2 = vec_perm(pix2v[0], pix2v[1], perm2);



        /* Calculate a sum of abs differences vector */

        t3 = vec_max(t1, t2);

        t4 = vec_min(t1, t2);

        t5 = vec_sub(t3, t4);



        /* Add each 4 pixel group together and put 4 results into sad */

        sad = vec_sum4s(t5, sad);



        pix1 += line_size;

        pix2 += line_size;

    }



    /* Sum up the four partial sums, and put the result into s */

    sumdiffs = vec_sums((vector signed int) sad, (vector signed int) zero);

    sumdiffs = vec_splat(sumdiffs, 3);

    vec_ste(sumdiffs, 0, &s);



    return s;

}
