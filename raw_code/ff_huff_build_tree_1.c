int ff_huff_build_tree(AVCodecContext *avctx, VLC *vlc, int nb_codes,

                       Node *nodes, huff_cmp_t cmp, int hnode_first)

{

    int i, j;

    int cur_node;

    int64_t sum = 0;



    for(i = 0; i < nb_codes; i++){

        nodes[i].sym = i;

        nodes[i].n0 = -2;

        sum += nodes[i].count;

    }



    if(sum >> 31) {

        av_log(avctx, AV_LOG_ERROR, "Too high symbol frequencies. Tree construction is not possible\n");

        return -1;

    }

    qsort(nodes, nb_codes, sizeof(Node), cmp);

    cur_node = nb_codes;


    for(i = 0; i < nb_codes*2-1; i += 2){

        nodes[cur_node].sym = HNODE;

        nodes[cur_node].count = nodes[i].count + nodes[i+1].count;

        nodes[cur_node].n0 = i;

        for(j = cur_node; j > 0; j--){

            if(nodes[j].count > nodes[j-1].count ||

               (nodes[j].count == nodes[j-1].count &&

                (!hnode_first || nodes[j].n0==j-1 || nodes[j].n0==j-2 ||

                 (nodes[j].sym!=HNODE && nodes[j-1].sym!=HNODE))))

                break;

            FFSWAP(Node, nodes[j], nodes[j-1]);

        }

        cur_node++;

    }

    if(build_huff_tree(vlc, nodes, nb_codes*2-2) < 0){

        av_log(avctx, AV_LOG_ERROR, "Error building tree\n");

        return -1;

    }

    return 0;

}