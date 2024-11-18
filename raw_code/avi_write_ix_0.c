static int avi_write_ix(AVFormatContext *s)

{

    AVIOContext *pb = s->pb;

    AVIContext *avi = s->priv_data;

    char tag[5];

    char ix_tag[] = "ix00";

    int i, j;



    assert(pb->seekable);



    if (avi->riff_id > AVI_MASTER_INDEX_SIZE)

        return -1;



    for (i = 0; i < s->nb_streams; i++) {

        AVIStream *avist = s->streams[i]->priv_data;

        int64_t ix, pos;



        avi_stream2fourcc(tag, i, s->streams[i]->codecpar->codec_type);

        ix_tag[3] = '0' + i;



        /* Writing AVI OpenDML leaf index chunk */

        ix = avio_tell(pb);

        ffio_wfourcc(pb, ix_tag);      /* ix?? */

        avio_wl32(pb, avist->indexes.entry * 8 + 24);

        /* chunk size */

        avio_wl16(pb, 2);           /* wLongsPerEntry */

        avio_w8(pb, 0);             /* bIndexSubType (0 == frame index) */

        avio_w8(pb, 1);             /* bIndexType (1 == AVI_INDEX_OF_CHUNKS) */

        avio_wl32(pb, avist->indexes.entry);

        /* nEntriesInUse */

        ffio_wfourcc(pb, tag);         /* dwChunkId */

        avio_wl64(pb, avi->movi_list); /* qwBaseOffset */

        avio_wl32(pb, 0);              /* dwReserved_3 (must be 0) */



        for (j = 0; j < avist->indexes.entry; j++) {

            AVIIentry *ie = avi_get_ientry(&avist->indexes, j);

            avio_wl32(pb, ie->pos + 8);

            avio_wl32(pb, ((uint32_t) ie->len & ~0x80000000) |

                          (ie->flags & 0x10 ? 0 : 0x80000000));

        }

        avio_flush(pb);

        pos = avio_tell(pb);



        /* Updating one entry in the AVI OpenDML master index */

        avio_seek(pb, avist->indexes.indx_start - 8, SEEK_SET);

        ffio_wfourcc(pb, "indx");             /* enabling this entry */

        avio_skip(pb, 8);

        avio_wl32(pb, avi->riff_id);          /* nEntriesInUse */

        avio_skip(pb, 16 * avi->riff_id);

        avio_wl64(pb, ix);                    /* qwOffset */

        avio_wl32(pb, pos - ix);              /* dwSize */

        avio_wl32(pb, avist->indexes.entry);  /* dwDuration */



        avio_seek(pb, pos, SEEK_SET);

    }

    return 0;

}
