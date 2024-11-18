static int wav_write_trailer(AVFormatContext *s)

{

    AVIOContext *pb  = s->pb;

    WAVMuxContext    *wav = s->priv_data;

    int64_t file_size, data_size;

    int64_t number_of_samples = 0;

    int rf64 = 0;



    avio_flush(pb);



    if (s->pb->seekable) {

        if (wav->write_peak != 2) {

            ff_end_tag(pb, wav->data);

            avio_flush(pb);

        }



        if (wav->write_peak && wav->peak_output) {

            peak_write_chunk(s);

            avio_flush(pb);

        }



        /* update file size */

        file_size = avio_tell(pb);

        data_size = file_size - wav->data;

        if (wav->rf64 == RF64_ALWAYS || (wav->rf64 == RF64_AUTO && file_size - 8 > UINT32_MAX)) {

            rf64 = 1;

        } else {

            avio_seek(pb, 4, SEEK_SET);

            avio_wl32(pb, (uint32_t)(file_size - 8));

            avio_seek(pb, file_size, SEEK_SET);



            avio_flush(pb);

        }



        number_of_samples = av_rescale(wav->maxpts - wav->minpts + wav->last_duration,

                                       s->streams[0]->codec->sample_rate * (int64_t)s->streams[0]->time_base.num,

                                       s->streams[0]->time_base.den);



        if(s->streams[0]->codec->codec_tag != 0x01) {

            /* Update num_samps in fact chunk */

            avio_seek(pb, wav->fact_pos, SEEK_SET);

            if (rf64 || (wav->rf64 == RF64_AUTO && number_of_samples > UINT32_MAX)) {

                rf64 = 1;

                avio_wl32(pb, -1);

            } else {

                avio_wl32(pb, number_of_samples);

                avio_seek(pb, file_size, SEEK_SET);

                avio_flush(pb);

            }

        }



        if (rf64) {

            /* overwrite RIFF with RF64 */

            avio_seek(pb, 0, SEEK_SET);

            ffio_wfourcc(pb, "RF64");

            avio_wl32(pb, -1);



            /* write ds64 chunk (overwrite JUNK if rf64 == RF64_AUTO) */

            avio_seek(pb, wav->ds64 - 8, SEEK_SET);

            ffio_wfourcc(pb, "ds64");

            avio_wl32(pb, 28);                  /* ds64 chunk size */

            avio_wl64(pb, file_size - 8);       /* RF64 chunk size */

            avio_wl64(pb, data_size);           /* data chunk size */

            avio_wl64(pb, number_of_samples);   /* fact chunk number of samples */

            avio_wl32(pb, 0);                   /* number of table entries for non-'data' chunks */



            /* write -1 in data chunk size */

            avio_seek(pb, wav->data - 4, SEEK_SET);

            avio_wl32(pb, -1);



            avio_seek(pb, file_size, SEEK_SET);

            avio_flush(pb);

        }

    }



    if (wav->write_peak)

        peak_free_buffers(s);



    return 0;

}
