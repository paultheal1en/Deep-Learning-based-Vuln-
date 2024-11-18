static uint32_t vmsvga_value_read(void *opaque, uint32_t address)

{

    uint32_t caps;

    struct vmsvga_state_s *s = opaque;

    switch (s->index) {

    case SVGA_REG_ID:

        return s->svgaid;



    case SVGA_REG_ENABLE:

        return s->enable;



    case SVGA_REG_WIDTH:

        return s->width;



    case SVGA_REG_HEIGHT:

        return s->height;



    case SVGA_REG_MAX_WIDTH:

        return SVGA_MAX_WIDTH;



    case SVGA_REG_MAX_HEIGHT:

        return SVGA_MAX_HEIGHT;



    case SVGA_REG_DEPTH:

        return s->depth;



    case SVGA_REG_BITS_PER_PIXEL:

        return (s->depth + 7) & ~7;



    case SVGA_REG_PSEUDOCOLOR:

        return 0x0;



    case SVGA_REG_RED_MASK:

        return s->wred;

    case SVGA_REG_GREEN_MASK:

        return s->wgreen;

    case SVGA_REG_BLUE_MASK:

        return s->wblue;



    case SVGA_REG_BYTES_PER_LINE:

        return ((s->depth + 7) >> 3) * s->new_width;



    case SVGA_REG_FB_START: {

        struct pci_vmsvga_state_s *pci_vmsvga

            = container_of(s, struct pci_vmsvga_state_s, chip);

        return pci_get_bar_addr(&pci_vmsvga->card, 1);

    }



    case SVGA_REG_FB_OFFSET:

        return 0x0;



    case SVGA_REG_VRAM_SIZE:

        return s->vga.vram_size;



    case SVGA_REG_FB_SIZE:

        return s->fb_size;



    case SVGA_REG_CAPABILITIES:

        caps = SVGA_CAP_NONE;

#ifdef HW_RECT_ACCEL

        caps |= SVGA_CAP_RECT_COPY;

#endif

#ifdef HW_FILL_ACCEL

        caps |= SVGA_CAP_RECT_FILL;

#endif

#ifdef HW_MOUSE_ACCEL

        if (dpy_cursor_define_supported(s->vga.ds)) {

            caps |= SVGA_CAP_CURSOR | SVGA_CAP_CURSOR_BYPASS_2 |

                    SVGA_CAP_CURSOR_BYPASS;

        }

#endif

        return caps;



    case SVGA_REG_MEM_START: {

        struct pci_vmsvga_state_s *pci_vmsvga

            = container_of(s, struct pci_vmsvga_state_s, chip);

        return pci_get_bar_addr(&pci_vmsvga->card, 2);

    }



    case SVGA_REG_MEM_SIZE:

        return s->fifo_size;



    case SVGA_REG_CONFIG_DONE:

        return s->config;



    case SVGA_REG_SYNC:

    case SVGA_REG_BUSY:

        return s->syncing;



    case SVGA_REG_GUEST_ID:

        return s->guest;



    case SVGA_REG_CURSOR_ID:

        return s->cursor.id;



    case SVGA_REG_CURSOR_X:

        return s->cursor.x;



    case SVGA_REG_CURSOR_Y:

        return s->cursor.x;



    case SVGA_REG_CURSOR_ON:

        return s->cursor.on;



    case SVGA_REG_HOST_BITS_PER_PIXEL:

        return (s->depth + 7) & ~7;



    case SVGA_REG_SCRATCH_SIZE:

        return s->scratch_size;



    case SVGA_REG_MEM_REGS:

    case SVGA_REG_NUM_DISPLAYS:

    case SVGA_REG_PITCHLOCK:

    case SVGA_PALETTE_BASE ... SVGA_PALETTE_END:

        return 0;



    default:

        if (s->index >= SVGA_SCRATCH_BASE &&

                s->index < SVGA_SCRATCH_BASE + s->scratch_size)

            return s->scratch[s->index - SVGA_SCRATCH_BASE];

        printf("%s: Bad register %02x\n", __FUNCTION__, s->index);

    }



    return 0;

}
