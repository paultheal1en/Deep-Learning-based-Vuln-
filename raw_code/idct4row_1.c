static inline void idct4row(DCTELEM *row)

{

    int c0, c1, c2, c3, a0, a1, a2, a3;

    //const uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;



    a0 = row[0];

    a1 = row[1];

    a2 = row[2];

    a3 = row[3];

    c0 = (a0 + a2)*R3 + (1 << (R_SHIFT - 1));

    c2 = (a0 - a2)*R3 + (1 << (R_SHIFT - 1));

    c1 = a1 * R1 + a3 * R2;

    c3 = a1 * R2 - a3 * R1;

    row[0]= (c0 + c1) >> R_SHIFT;

    row[1]= (c2 + c3) >> R_SHIFT;

    row[2]= (c2 - c3) >> R_SHIFT;

    row[3]= (c0 - c1) >> R_SHIFT;

}
