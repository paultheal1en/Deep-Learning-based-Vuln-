static void event_loop(VideoState *cur_stream)

{

    SDL_Event event;

    double incr, pos, frac;



    for(;;) {

        double x;

        SDL_WaitEvent(&event);

        switch(event.type) {

        case SDL_KEYDOWN:

            if (exit_on_keydown) {

                do_exit(cur_stream);

                break;

            }

            switch(event.key.keysym.sym) {

            case SDLK_ESCAPE:

            case SDLK_q:

                do_exit(cur_stream);

                break;

            case SDLK_f:

                toggle_full_screen(cur_stream);

                break;

            case SDLK_p:

            case SDLK_SPACE:

                if (cur_stream)

                    toggle_pause(cur_stream);

                break;

            case SDLK_s: //S: Step to next frame

                if (cur_stream)

                    step_to_next_frame(cur_stream);

                break;

            case SDLK_a:

                if (cur_stream)

                    stream_cycle_channel(cur_stream, AVMEDIA_TYPE_AUDIO);

                break;

            case SDLK_v:

                if (cur_stream)

                    stream_cycle_channel(cur_stream, AVMEDIA_TYPE_VIDEO);

                break;

            case SDLK_t:

                if (cur_stream)

                    stream_cycle_channel(cur_stream, AVMEDIA_TYPE_SUBTITLE);

                break;

            case SDLK_w:

                if (cur_stream)

                    toggle_audio_display(cur_stream);

                break;

            case SDLK_LEFT:

                incr = -10.0;

                goto do_seek;

            case SDLK_RIGHT:

                incr = 10.0;

                goto do_seek;

            case SDLK_UP:

                incr = 60.0;

                goto do_seek;

            case SDLK_DOWN:

                incr = -60.0;

            do_seek:

                if (cur_stream) {

                    if (seek_by_bytes) {

                        if (cur_stream->video_stream >= 0 && cur_stream->video_current_pos>=0){

                            pos= cur_stream->video_current_pos;

                        }else if(cur_stream->audio_stream >= 0 && cur_stream->audio_pkt.pos>=0){

                            pos= cur_stream->audio_pkt.pos;

                        }else

                            pos = avio_tell(cur_stream->ic->pb);

                        if (cur_stream->ic->bit_rate)

                            incr *= cur_stream->ic->bit_rate / 8.0;

                        else

                            incr *= 180000.0;

                        pos += incr;

                        stream_seek(cur_stream, pos, incr, 1);

                    } else {

                        pos = get_master_clock(cur_stream);

                        pos += incr;

                        stream_seek(cur_stream, (int64_t)(pos * AV_TIME_BASE), (int64_t)(incr * AV_TIME_BASE), 0);

                    }

                }

                break;

            default:

                break;

            }

            break;

        case SDL_MOUSEBUTTONDOWN:

            if (exit_on_mousedown) {

                do_exit(cur_stream);

                break;

            }

        case SDL_MOUSEMOTION:

            if(event.type ==SDL_MOUSEBUTTONDOWN){

                x= event.button.x;

            }else{

                if(event.motion.state != SDL_PRESSED)

                    break;

                x= event.motion.x;

            }

            if (cur_stream) {

                if(seek_by_bytes || cur_stream->ic->duration<=0){

                    uint64_t size=  avio_size(cur_stream->ic->pb);

                    stream_seek(cur_stream, size*x/cur_stream->width, 0, 1);

                }else{

                    int64_t ts;

                    int ns, hh, mm, ss;

                    int tns, thh, tmm, tss;

                    tns = cur_stream->ic->duration/1000000LL;

                    thh = tns/3600;

                    tmm = (tns%3600)/60;

                    tss = (tns%60);

                    frac = x/cur_stream->width;

                    ns = frac*tns;

                    hh = ns/3600;

                    mm = (ns%3600)/60;

                    ss = (ns%60);

                    fprintf(stderr, "Seek to %2.0f%% (%2d:%02d:%02d) of total duration (%2d:%02d:%02d)       \n", frac*100,

                            hh, mm, ss, thh, tmm, tss);

                    ts = frac*cur_stream->ic->duration;

                    if (cur_stream->ic->start_time != AV_NOPTS_VALUE)

                        ts += cur_stream->ic->start_time;

                    stream_seek(cur_stream, ts, 0, 0);

                }

            }

            break;

        case SDL_VIDEORESIZE:

            if (cur_stream) {

                screen = SDL_SetVideoMode(event.resize.w, event.resize.h, 0,

                                          SDL_HWSURFACE|SDL_RESIZABLE|SDL_ASYNCBLIT|SDL_HWACCEL);

                screen_width = cur_stream->width = event.resize.w;

                screen_height= cur_stream->height= event.resize.h;

            }

            break;

        case SDL_QUIT:

        case FF_QUIT_EVENT:

            do_exit(cur_stream);

            break;

        case FF_ALLOC_EVENT:

            video_open(event.user.data1);

            alloc_picture(event.user.data1);

            break;

        case FF_REFRESH_EVENT:

            video_refresh(event.user.data1);

            cur_stream->refresh=0;

            break;

        default:

            break;

        }

    }

}
