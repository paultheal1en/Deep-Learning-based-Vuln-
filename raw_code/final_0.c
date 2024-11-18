static void final(Real144_internal *glob, short *i1, short *i2, void *out,

                  int *statbuf, int len)

{

    int x, sum;

    int buffer[10];

    short *ptr;

    short *ptr2;



    memcpy(glob->work, statbuf,20);

    memcpy(glob->work + 10, i2, len * 2);



    buffer[9] = i1[0];

    buffer[8] = i1[1];

    buffer[7] = i1[2];

    buffer[6] = i1[3];

    buffer[5] = i1[4];

    buffer[4] = i1[5];

    buffer[3] = i1[6];

    buffer[2] = i1[7];

    buffer[1] = i1[8];

    buffer[0] = i1[9];



    ptr2 = (ptr = glob->work) + len;



    while (ptr < ptr2) {

        for(sum=0, x=0; x<=9; x++)

            sum += buffer[x] * (ptr[x]);



        sum = sum >> 12;

        x = ptr[10] - sum;



        if (x<-32768 || x>32767) {

            memset(out, 0, len * 2);

            memset(statbuf, 0, 20);

            return;

        }



        ptr[10] = x;

        ptr++;

    }

    memcpy(out, ptr+10 - len, len * 2);

    memcpy(statbuf, ptr, 20);

}
