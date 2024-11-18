static int mov_read_default(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    int64_t total_size = 0;

    MOVAtom a;

    int i;



    if (atom.size < 0)

        atom.size = INT64_MAX;

    while (total_size + 8 <= atom.size && !url_feof(pb)) {

        int (*parse)(MOVContext*, AVIOContext*, MOVAtom) = NULL;

        a.size = atom.size;

        a.type=0;

        if (atom.size >= 8) {

            a.size = avio_rb32(pb);

            a.type = avio_rl32(pb);

            total_size += 8;

            if (a.size == 1) { /* 64 bit extended size */

                a.size = avio_rb64(pb) - 8;

                total_size += 8;

            }

        }

        av_dlog(c->fc, "type: %08x '%.4s' parent:'%.4s' sz: %"PRId64" %"PRId64" %"PRId64"\n",

                a.type, (char*)&a.type, (char*)&atom.type, a.size, total_size, atom.size);

        if (a.size == 0) {

            a.size = atom.size - total_size + 8;

            if (a.size <= 8)

                break;

        }

        a.size -= 8;

        if (a.size < 0)

            break;

        a.size = FFMIN(a.size, atom.size - total_size);



        for (i = 0; mov_default_parse_table[i].type; i++)

            if (mov_default_parse_table[i].type == a.type) {

                parse = mov_default_parse_table[i].parse;

                break;

            }



        // container is user data

        if (!parse && (atom.type == MKTAG('u','d','t','a') ||

                       atom.type == MKTAG('i','l','s','t')))

            parse = mov_read_udta_string;



        if (!parse) { /* skip leaf atoms data */

            avio_skip(pb, a.size);

        } else {

            int64_t start_pos = avio_tell(pb);

            int64_t left;

            int err = parse(c, pb, a);

            if (err < 0)

                return err;

            if (c->found_moov && c->found_mdat &&

                (!pb->seekable || start_pos + a.size == avio_size(pb)))

                return 0;

            left = a.size - avio_tell(pb) + start_pos;

            if (left > 0) /* skip garbage at atom end */

                avio_skip(pb, left);

        }



        total_size += a.size;

    }



    if (total_size < atom.size && atom.size < 0x7ffff)

        avio_skip(pb, atom.size - total_size);



    return 0;

}
