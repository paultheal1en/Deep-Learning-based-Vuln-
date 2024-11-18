static int mov_read_udta_string(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    char tmp_key[5];

    char str[1024], key2[32], language[4] = {0};

    const char *key = NULL;

    uint16_t langcode = 0;

    uint32_t data_type = 0, str_size;

    int (*parse)(MOVContext*, AVIOContext*, unsigned, const char*) = NULL;



    switch (atom.type) {

    case MKTAG(0xa9,'n','a','m'): key = "title";     break;

    case MKTAG(0xa9,'a','u','t'):

    case MKTAG(0xa9,'A','R','T'): key = "artist";    break;

    case MKTAG( 'a','A','R','T'): key = "album_artist";    break;

    case MKTAG(0xa9,'w','r','t'): key = "composer";  break;

    case MKTAG( 'c','p','r','t'):

    case MKTAG(0xa9,'c','p','y'): key = "copyright"; break;

    case MKTAG(0xa9,'c','m','t'):

    case MKTAG(0xa9,'i','n','f'): key = "comment";   break;

    case MKTAG(0xa9,'a','l','b'): key = "album";     break;

    case MKTAG(0xa9,'d','a','y'): key = "date";      break;

    case MKTAG(0xa9,'g','e','n'): key = "genre";     break;

    case MKTAG( 'g','n','r','e'): key = "genre";

        parse = mov_metadata_gnre; break;

    case MKTAG(0xa9,'t','o','o'):

    case MKTAG(0xa9,'s','w','r'): key = "encoder";   break;

    case MKTAG(0xa9,'e','n','c'): key = "encoder";   break;

    case MKTAG(0xa9,'x','y','z'): key = "location";  break;

    case MKTAG( 'd','e','s','c'): key = "description";break;

    case MKTAG( 'l','d','e','s'): key = "synopsis";  break;

    case MKTAG( 't','v','s','h'): key = "show";      break;

    case MKTAG( 't','v','e','n'): key = "episode_id";break;

    case MKTAG( 't','v','n','n'): key = "network";   break;

    case MKTAG( 't','r','k','n'): key = "track";

        parse = mov_metadata_track_or_disc_number; break;

    case MKTAG( 'd','i','s','k'): key = "disc";

        parse = mov_metadata_track_or_disc_number; break;

    case MKTAG( 't','v','e','s'): key = "episode_sort";

        parse = mov_metadata_int8_bypass_padding; break;

    case MKTAG( 't','v','s','n'): key = "season_number";

        parse = mov_metadata_int8_bypass_padding; break;

    case MKTAG( 's','t','i','k'): key = "media_type";

        parse = mov_metadata_int8_no_padding; break;

    case MKTAG( 'h','d','v','d'): key = "hd_video";

        parse = mov_metadata_int8_no_padding; break;

    case MKTAG( 'p','g','a','p'): key = "gapless_playback";

        parse = mov_metadata_int8_no_padding; break;

    case MKTAG( 'l','o','c','i'):

        return mov_metadata_loci(c, pb, atom.size);

    }



    if (c->itunes_metadata && atom.size > 8) {

        int data_size = avio_rb32(pb);

        int tag = avio_rl32(pb);

        if (tag == MKTAG('d','a','t','a')) {

            data_type = avio_rb32(pb); // type

            avio_rb32(pb); // unknown

            str_size = data_size - 16;

            atom.size -= 16;



            if (atom.type == MKTAG('c', 'o', 'v', 'r')) {

                int ret = mov_read_covr(c, pb, data_type, str_size);

                if (ret < 0) {

                    av_log(c->fc, AV_LOG_ERROR, "Error parsing cover art.\n");

                    return ret;

                }

            }

        } else return 0;

    } else if (atom.size > 4 && key && !c->itunes_metadata) {

        str_size = avio_rb16(pb); // string length

        langcode = avio_rb16(pb);

        ff_mov_lang_to_iso639(langcode, language);

        atom.size -= 4;

    } else

        str_size = atom.size;



    if (c->export_all && !key) {

        snprintf(tmp_key, 5, "%.4s", (char*)&atom.type);

        key = tmp_key;

    }



    if (!key)

        return 0;

    if (atom.size < 0)

        return AVERROR_INVALIDDATA;



    str_size = FFMIN3(sizeof(str)-1, str_size, atom.size);



    if (parse)

        parse(c, pb, str_size, key);

    else {

        if (data_type == 3 || (data_type == 0 && (langcode < 0x400 || langcode == 0x7fff))) { // MAC Encoded

            mov_read_mac_string(c, pb, str_size, str, sizeof(str));

        } else {

            avio_read(pb, str, str_size);

            str[str_size] = 0;

        }

        c->fc->event_flags |= AVFMT_EVENT_FLAG_METADATA_UPDATED;

        av_dict_set(&c->fc->metadata, key, str, 0);

        if (*language && strcmp(language, "und")) {

            snprintf(key2, sizeof(key2), "%s-%s", key, language);

            av_dict_set(&c->fc->metadata, key2, str, 0);

        }

    }

    av_dlog(c->fc, "lang \"%3s\" ", language);

    av_dlog(c->fc, "tag \"%s\" value \"%s\" atom \"%.4s\" %d %"PRId64"\n",

            key, str, (char*)&atom.type, str_size, atom.size);



    return 0;

}
