void event_loop(void)

{

    SDL_Event event;

    double incr, pos, frac;



    for(;;) {

        SDL_WaitEvent(&event);

        switch(event.type) {

        case SDL_KEYDOWN:

            switch(event.key.keysym.sym) {

            case SDLK_ESCAPE:

            case SDLK_q:

                do_exit();

                break;

            case SDLK_f:

                toggle_full_screen();

                break;

            case SDLK_p:

            case SDLK_SPACE:

                toggle_pause();

                break;

            case SDLK_s: //S: Step to next frame

                step_to_next_frame();

                break;

            case SDLK_a:

                if (cur_stream) 

                    stream_cycle_channel(cur_stream, CODEC_TYPE_AUDIO);

                break;

            case SDLK_v:

                if (cur_stream) 

                    stream_cycle_channel(cur_stream, CODEC_TYPE_VIDEO);

                break;

            case SDLK_w:

                toggle_audio_display();

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

                    pos = get_master_clock(cur_stream);

printf("%f %f %d %d %d %d\n", (float)pos, (float)incr, cur_stream->av_sync_type == AV_SYNC_VIDEO_MASTER, 

cur_stream->av_sync_type == AV_SYNC_AUDIO_MASTER, cur_stream->video_st, cur_stream->audio_st);

                    pos += incr;

                    stream_seek(cur_stream, (int64_t)(pos * AV_TIME_BASE));

                }

                break;

            default:

                break;

            }

            break;

        case SDL_MOUSEBUTTONDOWN:

	    if (cur_stream) {

		int ns, hh, mm, ss;

		int tns, thh, tmm, tss;

		tns = cur_stream->ic->duration/1000000LL;

		thh = tns/3600;

		tmm = (tns%3600)/60;

		tss = (tns%60);

		frac = (double)event.button.x/(double)cur_stream->width;

		ns = frac*tns;

		hh = ns/3600;

		mm = (ns%3600)/60;

		ss = (ns%60);

		fprintf(stderr, "Seek to %2.0f%% (%2d:%02d:%02d) of total duration (%2d:%02d:%02d)       \n", frac*100,

			hh, mm, ss, thh, tmm, tss);

		stream_seek(cur_stream, (int64_t)(cur_stream->ic->start_time+frac*cur_stream->ic->duration));

	    }

	    break;

        case SDL_VIDEORESIZE:

            if (cur_stream) {

                screen = SDL_SetVideoMode(event.resize.w, event.resize.h, 0, 

                                          SDL_HWSURFACE|SDL_RESIZABLE|SDL_ASYNCBLIT|SDL_HWACCEL);

                cur_stream->width = event.resize.w;

                cur_stream->height = event.resize.h;

            }

            break;

        case SDL_QUIT:

        case FF_QUIT_EVENT:

            do_exit();

            break;

        case FF_ALLOC_EVENT:

            alloc_picture(event.user.data1);

            break;

        case FF_REFRESH_EVENT:

            video_refresh_timer(event.user.data1);

            break;

        default:

            break;

        }

    }

}
