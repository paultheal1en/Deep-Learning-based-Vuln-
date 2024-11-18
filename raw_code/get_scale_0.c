static inline int get_scale(GetBitContext *gb, int level, int value)

{

    if (level < 5) {

        /* huffman encoded */

        value += get_bitalloc(gb, &dca_scalefactor, level);

    } else if (level < 8)

        value = get_bits(gb, level + 1);

    return value;

}
