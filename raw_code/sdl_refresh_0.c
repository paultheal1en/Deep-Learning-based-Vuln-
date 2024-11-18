static void sdl_refresh(DisplayState *ds)

{

    SDL_Event ev1, *ev = &ev1;

    int mod_state;

    int buttonstate = SDL_GetMouseState(NULL, NULL);



    if (last_vm_running != vm_running) {

        last_vm_running = vm_running;

        sdl_update_caption();

    }



    vga_hw_update();

    SDL_EnableUNICODE(!is_graphic_console());



    while (SDL_PollEvent(ev)) {

        switch (ev->type) {

        case SDL_VIDEOEXPOSE:

            sdl_update(ds, 0, 0, screen->w, screen->h);

            break;

        case SDL_KEYDOWN:

        case SDL_KEYUP:

            if (ev->type == SDL_KEYDOWN) {

                if (!alt_grab) {

                    mod_state = (SDL_GetModState() & gui_grab_code) ==

                                gui_grab_code;

                } else {

                    mod_state = (SDL_GetModState() & (gui_grab_code | KMOD_LSHIFT)) ==

                                (gui_grab_code | KMOD_LSHIFT);

                }

                gui_key_modifier_pressed = mod_state;

                if (gui_key_modifier_pressed) {

                    int keycode;

                    keycode = sdl_keyevent_to_keycode(&ev->key);

                    switch(keycode) {

                    case 0x21: /* 'f' key on US keyboard */

                        toggle_full_screen(ds);

                        gui_keysym = 1;

                        break;

                    case 0x02 ... 0x0a: /* '1' to '9' keys */

                        /* Reset the modifiers sent to the current console */

                        reset_keys();

                        console_select(keycode - 0x02);

                        if (!is_graphic_console()) {

                            /* display grab if going to a text console */

                            if (gui_grab)

                                sdl_grab_end();

                        }

                        gui_keysym = 1;

                        break;

                    default:

                        break;

                    }

                } else if (!is_graphic_console()) {

                    int keysym;

                    keysym = 0;

                    if (ev->key.keysym.mod & (KMOD_LCTRL | KMOD_RCTRL)) {

                        switch(ev->key.keysym.sym) {

                        case SDLK_UP: keysym = QEMU_KEY_CTRL_UP; break;

                        case SDLK_DOWN: keysym = QEMU_KEY_CTRL_DOWN; break;

                        case SDLK_LEFT: keysym = QEMU_KEY_CTRL_LEFT; break;

                        case SDLK_RIGHT: keysym = QEMU_KEY_CTRL_RIGHT; break;

                        case SDLK_HOME: keysym = QEMU_KEY_CTRL_HOME; break;

                        case SDLK_END: keysym = QEMU_KEY_CTRL_END; break;

                        case SDLK_PAGEUP: keysym = QEMU_KEY_CTRL_PAGEUP; break;

                        case SDLK_PAGEDOWN: keysym = QEMU_KEY_CTRL_PAGEDOWN; break;

                        default: break;

                        }

                    } else {

                        switch(ev->key.keysym.sym) {

                        case SDLK_UP: keysym = QEMU_KEY_UP; break;

                        case SDLK_DOWN: keysym = QEMU_KEY_DOWN; break;

                        case SDLK_LEFT: keysym = QEMU_KEY_LEFT; break;

                        case SDLK_RIGHT: keysym = QEMU_KEY_RIGHT; break;

                        case SDLK_HOME: keysym = QEMU_KEY_HOME; break;

                        case SDLK_END: keysym = QEMU_KEY_END; break;

                        case SDLK_PAGEUP: keysym = QEMU_KEY_PAGEUP; break;

                        case SDLK_PAGEDOWN: keysym = QEMU_KEY_PAGEDOWN; break;

                        case SDLK_BACKSPACE: keysym = QEMU_KEY_BACKSPACE; break;

                        case SDLK_DELETE: keysym = QEMU_KEY_DELETE; break;

                        default: break;

                        }

                    }

                    if (keysym) {

                        kbd_put_keysym(keysym);

                    } else if (ev->key.keysym.unicode != 0) {

                        kbd_put_keysym(ev->key.keysym.unicode);

                    }

                }

            } else if (ev->type == SDL_KEYUP) {

                if (!alt_grab) {

                    mod_state = (ev->key.keysym.mod & gui_grab_code);

                } else {

                    mod_state = (ev->key.keysym.mod &

                                 (gui_grab_code | KMOD_LSHIFT));

                }

                if (!mod_state) {

                    if (gui_key_modifier_pressed) {

                        gui_key_modifier_pressed = 0;

                        if (gui_keysym == 0) {

                            /* exit/enter grab if pressing Ctrl-Alt */

                            if (!gui_grab) {

                                /* if the application is not active,

                                   do not try to enter grab state. It

                                   prevents

                                   'SDL_WM_GrabInput(SDL_GRAB_ON)'

                                   from blocking all the application

                                   (SDL bug). */

                                if (SDL_GetAppState() & SDL_APPACTIVE)

                                    sdl_grab_start();

                            } else {

                                sdl_grab_end();

                            }

                            /* SDL does not send back all the

                               modifiers key, so we must correct it */

                            reset_keys();

                            break;

                        }

                        gui_keysym = 0;

                    }

                }

            }

            if (is_graphic_console() && !gui_keysym)

                sdl_process_key(&ev->key);

            break;

        case SDL_QUIT:

            if (!no_quit) {

                qemu_system_shutdown_request();

                vm_start();	/* In case we're paused */

            }

            break;

        case SDL_MOUSEMOTION:

            if (gui_grab || kbd_mouse_is_absolute() ||

                absolute_enabled) {

                sdl_send_mouse_event(ev->motion.xrel, ev->motion.yrel, 0,

                       ev->motion.x, ev->motion.y, ev->motion.state);

            }

            break;

        case SDL_MOUSEBUTTONDOWN:

        case SDL_MOUSEBUTTONUP:

            {

                SDL_MouseButtonEvent *bev = &ev->button;

                if (!gui_grab && !kbd_mouse_is_absolute()) {

                    if (ev->type == SDL_MOUSEBUTTONDOWN &&

                        (bev->button == SDL_BUTTON_LEFT)) {

                        /* start grabbing all events */

                        sdl_grab_start();

                    }

                } else {

                    int dz;

                    dz = 0;

                    if (ev->type == SDL_MOUSEBUTTONDOWN) {

                        buttonstate |= SDL_BUTTON(bev->button);

                    } else {

                        buttonstate &= ~SDL_BUTTON(bev->button);

                    }

#ifdef SDL_BUTTON_WHEELUP

                    if (bev->button == SDL_BUTTON_WHEELUP && ev->type == SDL_MOUSEBUTTONDOWN) {

                        dz = -1;

                    } else if (bev->button == SDL_BUTTON_WHEELDOWN && ev->type == SDL_MOUSEBUTTONDOWN) {

                        dz = 1;

                    }

#endif

                    sdl_send_mouse_event(0, 0, dz, bev->x, bev->y, buttonstate);

                }

            }

            break;

        case SDL_ACTIVEEVENT:

            if (gui_grab && ev->active.state == SDL_APPINPUTFOCUS &&

                !ev->active.gain && !gui_fullscreen_initial_grab) {

                sdl_grab_end();

            }

            if (ev->active.state & SDL_APPACTIVE) {

                if (ev->active.gain) {

                    /* Back to default interval */

                    ds->gui_timer_interval = 0;

                } else {

                    /* Sleeping interval */

                    ds->gui_timer_interval = 500;

                }

            }

            break;

        default:

            break;

        }

    }

}
