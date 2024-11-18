static void xbr2x(AVFrame * input, AVFrame * output, const uint32_t * r2y)

{

    int x,y;

    int next_line = output->linesize[0]>>2;



    for (y = 0; y < input->height; y++) {



        uint32_t pprev;

        uint32_t pprev2;



        uint32_t * E = (uint32_t *)(output->data[0] + y * output->linesize[0] * 2);



        /* middle. Offset of -8 is given */

        uint32_t * sa2 = (uint32_t *)(input->data[0] + y * input->linesize[0] - 8);

        /* up one */

        uint32_t * sa1 = sa2 - (input->linesize[0]>>2);

        /* up two */

        uint32_t * sa0 = sa1 - (input->linesize[0]>>2);

        /* down one */

        uint32_t * sa3 = sa2 + (input->linesize[0]>>2);

        /* down two */

        uint32_t * sa4 = sa3 + (input->linesize[0]>>2);



        if (y <= 1) {

            sa0 = sa1;

            if (y == 0) {

                sa0 = sa1 = sa2;

            }

        }



        if (y >= input->height - 2) {

            sa4 = sa3;

            if (y == input->height - 1) {

                sa4 = sa3 = sa2;

            }

        }



        pprev = pprev2 = 2;



        for (x = 0; x < input->width; x++) {

            uint32_t B1 = sa0[2];

            uint32_t PB = sa1[2];

            uint32_t PE = sa2[2];

            uint32_t PH = sa3[2];

            uint32_t H5 = sa4[2];



            uint32_t A1 = sa0[pprev];

            uint32_t PA = sa1[pprev];

            uint32_t PD = sa2[pprev];

            uint32_t PG = sa3[pprev];

            uint32_t G5 = sa4[pprev];



            uint32_t A0 = sa1[pprev2];

            uint32_t D0 = sa2[pprev2];

            uint32_t G0 = sa3[pprev2];



            uint32_t C1 = 0;

            uint32_t PC = 0;

            uint32_t PF = 0;

            uint32_t PI = 0;

            uint32_t I5 = 0;



            uint32_t C4 = 0;

            uint32_t F4 = 0;

            uint32_t I4 = 0;



            if (x >= input->width - 2) {

                if (x == input->width - 1) {

                    C1 = sa0[2];

                    PC = sa1[2];

                    PF = sa2[2];

                    PI = sa3[2];

                    I5 = sa4[2];



                    C4 = sa1[2];

                    F4 = sa2[2];

                    I4 = sa3[2];

                } else {

                    C1 = sa0[3];

                    PC = sa1[3];

                    PF = sa2[3];

                    PI = sa3[3];

                    I5 = sa4[3];



                    C4 = sa1[3];

                    F4 = sa2[3];

                    I4 = sa3[3];

                }

            } else {

                C1 = sa0[3];

                PC = sa1[3];

                PF = sa2[3];

                PI = sa3[3];

                I5 = sa4[3];



                C4 = sa1[4];

                F4 = sa2[4];

                I4 = sa3[4];

            }



            E[0] = E[1] = E[next_line] = E[next_line + 1] = PE; // 0, 1, 2, 3



            FILT2(PE, PI, PH, PF, PG, PC, PD, PB, PA, G5, C4, G0, D0, C1, B1, F4, I4, H5, I5, A0, A1, 0, 1, next_line, next_line+1);

            FILT2(PE, PC, PF, PB, PI, PA, PH, PD, PG, I4, A1, I5, H5, A0, D0, B1, C1, F4, C4, G5, G0, next_line, 0, next_line+1, 1);

            FILT2(PE, PA, PB, PD, PC, PG, PF, PH, PI, C1, G0, C4, F4, G5, H5, D0, A0, B1, A1, I4, I5, next_line+1, next_line, 1, 0);

            FILT2(PE, PG, PD, PH, PA, PI, PB, PF, PC, A0, I5, A1, B1, I4, F4, H5, G5, D0, G0, C1, C4, 1, next_line+1, 0, next_line);



            sa0 += 1;

            sa1 += 1;

            sa2 += 1;

            sa3 += 1;

            sa4 += 1;



            E += 2;



            if (pprev2){

                pprev2--;

                pprev = 1;

            }

        }

    }

}
