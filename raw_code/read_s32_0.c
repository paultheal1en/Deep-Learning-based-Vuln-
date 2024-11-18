static int32_t read_s32(uint8_t *data, size_t offset)

{

    return (int32_t)((data[offset] << 24) | (data[offset + 1] << 16) |

		     (data[offset + 2] << 8) | data[offset + 3]);

}
