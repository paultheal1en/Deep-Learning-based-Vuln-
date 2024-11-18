static inline int signed_shift(int i, int shift) {

    if (shift > 0)

        return i << shift;

    return i >> -shift;

}
