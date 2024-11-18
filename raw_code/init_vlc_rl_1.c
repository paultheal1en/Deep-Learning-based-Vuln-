void init_vlc_rl(RLTable *rl)

{

    int i, q;

    

    init_vlc(&rl->vlc, 9, rl->n + 1, 

             &rl->table_vlc[0][1], 4, 2,

             &rl->table_vlc[0][0], 4, 2);



    

    for(q=0; q<32; q++){

        int qmul= q*2;

        int qadd= (q-1)|1;

        

        if(q==0){

            qmul=1;

            qadd=0;

        }

        

        rl->rl_vlc[q]= av_malloc(rl->vlc.table_size*sizeof(RL_VLC_ELEM));

        for(i=0; i<rl->vlc.table_size; i++){

            int code= rl->vlc.table[i][0];

            int len = rl->vlc.table[i][1];

            int level, run;

        

            if(len==0){ // illegal code

                run= 66;

                level= MAX_LEVEL;

            }else if(len<0){ //more bits needed

                run= 0;

                level= code;

            }else{

                if(code==rl->n){ //esc

                    run= 66;

                    level= 0;

                }else{

                    run=   rl->table_run  [code] + 1;

                    level= rl->table_level[code] * qmul + qadd;

                    if(code >= rl->last) run+=192;

                }

            }

            rl->rl_vlc[q][i].len= len;

            rl->rl_vlc[q][i].level= level;

            rl->rl_vlc[q][i].run= run;

        }

    }

}
