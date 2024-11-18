static int matroska_parse_cluster(MatroskaDemuxContext *matroska)

{

    MatroskaCluster cluster = { 0 };

    EbmlList *blocks_list;

    MatroskaBlock *blocks;

    int i, res;

    int64_t pos = url_ftell(matroska->ctx->pb);

    matroska->prev_pkt = NULL;

    if (matroska->has_cluster_id){

        /* For the first cluster we parse, its ID was already read as

           part of matroska_read_header(), so don't read it again */

        res = ebml_parse_id(matroska, matroska_clusters,

                            MATROSKA_ID_CLUSTER, &cluster);

        pos -= 4;  /* sizeof the ID which was already read */

        matroska->has_cluster_id = 0;

    } else

        res = ebml_parse(matroska, matroska_clusters, &cluster);

    blocks_list = &cluster.blocks;

    blocks = blocks_list->elem;

    for (i=0; i<blocks_list->nb_elem; i++)

        if (blocks[i].bin.size > 0) {

            int is_keyframe = blocks[i].non_simple ? !blocks[i].reference : -1;

            res=matroska_parse_block(matroska,

                                     blocks[i].bin.data, blocks[i].bin.size,

                                     blocks[i].bin.pos,  cluster.timecode,

                                     blocks[i].duration, is_keyframe,

                                     pos);

        }

    ebml_free(matroska_cluster, &cluster);

    if (res < 0)  matroska->done = 1;

    return res;

}
