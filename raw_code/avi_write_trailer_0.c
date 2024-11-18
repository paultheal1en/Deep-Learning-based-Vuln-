static int avi_write_trailer(AVFormatContext *s)

{

    AVIContext *avi = s->priv_data;

    AVIOContext *pb = s->pb;

    int res = 0;

    int i, j, n, nb_frames;

    int64_t file_size;



    if (pb->seekable) {

        if (avi->riff_id == 1) {

            ff_end_tag(pb, avi->movi_list);

            res = avi_write_idx1(s);

            ff_end_tag(pb, avi->riff_start);

        } else {

            avi_write_ix(s);

            ff_end_tag(pb, avi->movi_list);

            ff_end_tag(pb, avi->riff_start);



            file_size = avio_tell(pb);

            avio_seek(pb, avi->odml_list - 8, SEEK_SET);

            ffio_wfourcc(pb, "LIST"); /* Making this AVI OpenDML one */

            avio_skip(pb, 16);



            for (n = nb_frames = 0; n < s->nb_streams; n++) {

                AVCodecParameters *par = s->streams[n]->codecpar;

                AVIStream *avist       = s->streams[n]->priv_data;



                if (par->codec_type == AVMEDIA_TYPE_VIDEO) {

                    if (nb_frames < avist->packet_count)

                        nb_frames = avist->packet_count;

                } else {

                    if (par->codec_id == AV_CODEC_ID_MP2 ||

                        par->codec_id == AV_CODEC_ID_MP3)

                        nb_frames += avist->packet_count;

                }

            }

            avio_wl32(pb, nb_frames);

            avio_seek(pb, file_size, SEEK_SET);



            avi_write_counters(s, avi->riff_id);

        }

    }



    for (i = 0; i < s->nb_streams; i++) {

        AVIStream *avist = s->streams[i]->priv_data;

        for (j = 0; j < avist->indexes.ents_allocated / AVI_INDEX_CLUSTER_SIZE; j++)

            av_free(avist->indexes.cluster[j]);

        av_freep(&avist->indexes.cluster);

        avist->indexes.ents_allocated = avist->indexes.entry = 0;

    }



    return res;

}
