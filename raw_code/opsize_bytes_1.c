static inline int opsize_bytes(int opsize)

{

    switch (opsize) {

    case OS_BYTE: return 1;

    case OS_WORD: return 2;

    case OS_LONG: return 4;

    case OS_SINGLE: return 4;

    case OS_DOUBLE: return 8;

    default:

        qemu_assert(0, "bad operand size");

        return 0;

    }

}
