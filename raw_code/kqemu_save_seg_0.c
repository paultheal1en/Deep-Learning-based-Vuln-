static inline void kqemu_save_seg(SegmentCache *sc,

                                  const struct kqemu_segment_cache *ksc)

{

    sc->selector = ksc->selector;

    sc->flags = ksc->flags;

    sc->limit = ksc->limit;

    sc->base = ksc->base;

}
