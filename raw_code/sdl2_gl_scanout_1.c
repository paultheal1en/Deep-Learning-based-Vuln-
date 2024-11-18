void sdl2_gl_scanout(DisplayChangeListener *dcl,

                     uint32_t backing_id, bool backing_y_0_top,


                     uint32_t x, uint32_t y,

                     uint32_t w, uint32_t h)

{

    struct sdl2_console *scon = container_of(dcl, struct sdl2_console, dcl);



    assert(scon->opengl);

    scon->x = x;

    scon->y = y;

    scon->w = w;

    scon->h = h;

    scon->tex_id = backing_id;

    scon->y0_top = backing_y_0_top;



    SDL_GL_MakeCurrent(scon->real_window, scon->winctx);



    if (scon->tex_id == 0 || scon->w == 0 || scon->h == 0) {

        sdl2_set_scanout_mode(scon, false);

        return;

    }



    sdl2_set_scanout_mode(scon, true);

    if (!scon->fbo_id) {

        glGenFramebuffers(1, &scon->fbo_id);

    }



    glBindFramebuffer(GL_FRAMEBUFFER_EXT, scon->fbo_id);

    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,

                              GL_TEXTURE_2D, scon->tex_id, 0);

}