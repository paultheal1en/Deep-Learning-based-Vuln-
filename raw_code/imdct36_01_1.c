static void imdct36(int *out, int *buf, int *in, int *win)

{

    int i, j, t0, t1, t2, t3, s0, s1, s2, s3;

    int tmp[18], *tmp1, *in1;



    for(i=17;i>=1;i--)

        in[i] += in[i-1];

    for(i=17;i>=3;i-=2)

        in[i] += in[i-2];



    for(j=0;j<2;j++) {

        tmp1 = tmp + j;

        in1 = in + j;

#if 0

//more accurate but slower

        int64_t t0, t1, t2, t3;

        t2 = in1[2*4] + in1[2*8] - in1[2*2];

        

        t3 = (in1[2*0] + (int64_t)(in1[2*6]>>1))<<32;

        t1 = in1[2*0] - in1[2*6];

        tmp1[ 6] = t1 - (t2>>1);

        tmp1[16] = t1 + t2;



        t0 = MUL64(2*(in1[2*2] + in1[2*4]),    C2);

        t1 = MUL64(   in1[2*4] - in1[2*8] , -2*C8);

        t2 = MUL64(2*(in1[2*2] + in1[2*8]),   -C4);

        

        tmp1[10] = (t3 - t0 - t2) >> 32;

        tmp1[ 2] = (t3 + t0 + t1) >> 32;

        tmp1[14] = (t3 + t2 - t1) >> 32;

        

        tmp1[ 4] = MULH(2*(in1[2*5] + in1[2*7] - in1[2*1]), -C3);

        t2 = MUL64(2*(in1[2*1] + in1[2*5]),    C1);

        t3 = MUL64(   in1[2*5] - in1[2*7] , -2*C7);

        t0 = MUL64(2*in1[2*3], C3);



        t1 = MUL64(2*(in1[2*1] + in1[2*7]),   -C5);



        tmp1[ 0] = (t2 + t3 + t0) >> 32;

        tmp1[12] = (t2 + t1 - t0) >> 32;

        tmp1[ 8] = (t3 - t1 - t0) >> 32;

#else

        t2 = in1[2*4] + in1[2*8] - in1[2*2];

        

        t3 = in1[2*0] + (in1[2*6]>>1);

        t1 = in1[2*0] - in1[2*6];

        tmp1[ 6] = t1 - (t2>>1);

        tmp1[16] = t1 + t2;



        t0 = MULH(2*(in1[2*2] + in1[2*4]),    C2);

        t1 = MULH(   in1[2*4] - in1[2*8] , -2*C8);

        t2 = MULH(2*(in1[2*2] + in1[2*8]),   -C4);

        

        tmp1[10] = t3 - t0 - t2;

        tmp1[ 2] = t3 + t0 + t1;

        tmp1[14] = t3 + t2 - t1;

        

        tmp1[ 4] = MULH(2*(in1[2*5] + in1[2*7] - in1[2*1]), -C3);

        t2 = MULH(2*(in1[2*1] + in1[2*5]),    C1);

        t3 = MULH(   in1[2*5] - in1[2*7] , -2*C7);

        t0 = MULH(2*in1[2*3], C3);



        t1 = MULH(2*(in1[2*1] + in1[2*7]),   -C5);



        tmp1[ 0] = t2 + t3 + t0;

        tmp1[12] = t2 + t1 - t0;

        tmp1[ 8] = t3 - t1 - t0;

#endif

    }



    i = 0;

    for(j=0;j<4;j++) {

        t0 = tmp[i];

        t1 = tmp[i + 2];

        s0 = t1 + t0;

        s2 = t1 - t0;



        t2 = tmp[i + 1];

        t3 = tmp[i + 3];

        s1 = MULL(t3 + t2, icos36[j]);

        s3 = MULL(t3 - t2, icos36[8 - j]);

        

        t0 = (s0 + s1) << 5;

        t1 = (s0 - s1) << 5;

        out[(9 + j)*SBLIMIT] =  MULH(t1, win[9 + j]) + buf[9 + j];

        out[(8 - j)*SBLIMIT] =  MULH(t1, win[8 - j]) + buf[8 - j];

        buf[9 + j] = MULH(t0, win[18 + 9 + j]);

        buf[8 - j] = MULH(t0, win[18 + 8 - j]);

        

        t0 = (s2 + s3) << 5;

        t1 = (s2 - s3) << 5;

        out[(9 + 8 - j)*SBLIMIT] =  MULH(t1, win[9 + 8 - j]) + buf[9 + 8 - j];

        out[(        j)*SBLIMIT] =  MULH(t1, win[        j]) + buf[        j];

        buf[9 + 8 - j] = MULH(t0, win[18 + 9 + 8 - j]);

        buf[      + j] = MULH(t0, win[18         + j]);

        i += 4;

    }



    s0 = tmp[16];

    s1 = MULL(tmp[17], icos36[4]);

    t0 = (s0 + s1) << 5;

    t1 = (s0 - s1) << 5;

    out[(9 + 4)*SBLIMIT] =  MULH(t1, win[9 + 4]) + buf[9 + 4];

    out[(8 - 4)*SBLIMIT] =  MULH(t1, win[8 - 4]) + buf[8 - 4];

    buf[9 + 4] = MULH(t0, win[18 + 9 + 4]);

    buf[8 - 4] = MULH(t0, win[18 + 8 - 4]);

}
