// if having troubles with min/max, uncomment the following
// #undef min
// #undef max

#ifdef __has_attribute
#define HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define HAVE_ATTRIBUTE(x) 0
#endif
#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))
#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(4)))
#else
#define DATA_ALIGN_ATTRIBUTE
#endif

const unsigned char exercisemodel[] = {
  0x20, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x12, 0x00, 0x1c, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0c, 0x00,
  0x10, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x00, 0x12, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x90, 0x11, 0x00, 0x00, 0xb0, 0x05, 0x00, 0x00,
  0x98, 0x05, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0c, 0x00,
  0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
  0x1b, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 0x6d, 0x69, 0x6e, 0x5f,
  0x72, 0x75, 0x6e, 0x74, 0x69, 0x6d, 0x65, 0x5f, 0x76, 0x65, 0x72, 0x73,
  0x69, 0x6f, 0x6e, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x48, 0x05, 0x00, 0x00,
  0x34, 0x05, 0x00, 0x00, 0x20, 0x05, 0x00, 0x00, 0x0c, 0x05, 0x00, 0x00,
  0xe0, 0x04, 0x00, 0x00, 0xbc, 0x04, 0x00, 0x00, 0x98, 0x04, 0x00, 0x00,
  0x74, 0x04, 0x00, 0x00, 0x50, 0x04, 0x00, 0x00, 0x2c, 0x04, 0x00, 0x00,
  0x08, 0x04, 0x00, 0x00, 0x94, 0x03, 0x00, 0x00, 0x20, 0x03, 0x00, 0x00,
  0xac, 0x02, 0x00, 0x00, 0x48, 0x02, 0x00, 0x00, 0xe4, 0x01, 0x00, 0x00,
  0x80, 0x01, 0x00, 0x00, 0xdc, 0x00, 0x00, 0x00, 0xd0, 0x00, 0x00, 0x00,
  0xbc, 0x00, 0x00, 0x00, 0xa8, 0x00, 0x00, 0x00, 0x94, 0x00, 0x00, 0x00,
  0x80, 0x00, 0x00, 0x00, 0x6c, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00,
  0x44, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x0a, 0xfa, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x31, 0x2e, 0x35, 0x2e, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x74, 0xef, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x84, 0xef, 0xff, 0xff,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x94, 0xef, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xa4, 0xef, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xb4, 0xef, 0xff, 0xff,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xc4, 0xef, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xd4, 0xef, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe4, 0xef, 0xff, 0xff,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xf4, 0xef, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0xba, 0xfa, 0xff, 0xff,
  0x04, 0x00, 0x00, 0x00, 0x90, 0x00, 0x00, 0x00, 0x32, 0xeb, 0x26, 0xc0,
  0xe9, 0xf9, 0x4c, 0xbf, 0x37, 0x49, 0x69, 0x3f, 0x49, 0xa2, 0x8e, 0x3f,
  0xce, 0xe1, 0xcb, 0x3e, 0xb6, 0x95, 0x10, 0x3f, 0x88, 0xfe, 0x6b, 0x3e,
  0x48, 0x8f, 0x49, 0xbe, 0xcb, 0x4c, 0x3e, 0xc0, 0xe0, 0x55, 0x8b, 0xbc,
  0xd0, 0xe7, 0x9b, 0x3f, 0x62, 0x02, 0xb9, 0xbe, 0xac, 0x16, 0xba, 0x3d,
  0x6f, 0x55, 0x5c, 0xbe, 0x67, 0x2d, 0x51, 0x3f, 0xb0, 0x3d, 0xfa, 0x3d,
  0x54, 0xf3, 0x40, 0xbe, 0x56, 0x51, 0x05, 0xbf, 0x44, 0xb8, 0x01, 0x3e,
  0x40, 0x2e, 0xc1, 0x3c, 0x55, 0x5b, 0x09, 0xbf, 0x45, 0x40, 0x8a, 0xbe,
  0xe9, 0xef, 0xb1, 0xbe, 0xaa, 0x4f, 0x10, 0x3e, 0x87, 0x53, 0x22, 0x40,
  0xb7, 0x5b, 0x78, 0x3e, 0x07, 0x50, 0x68, 0xbf, 0x46, 0x6a, 0xac, 0xbf,
  0x57, 0x0f, 0x8b, 0xbe, 0x61, 0x32, 0xe3, 0xbe, 0xba, 0x36, 0x27, 0xbe,
  0x35, 0x04, 0x20, 0x3f, 0xd4, 0x52, 0x1f, 0x40, 0x9b, 0x52, 0x13, 0x3f,
  0x38, 0x38, 0xa2, 0xbf, 0x2f, 0xc3, 0x3d, 0xbe, 0x00, 0x00, 0x00, 0x00,
  0x5a, 0xfb, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00,
  0xe1, 0x61, 0xbc, 0x3f, 0x01, 0xbe, 0xa5, 0xbf, 0xe8, 0xac, 0x97, 0xbd,
  0xae, 0xbe, 0xa5, 0xbe, 0xa7, 0x47, 0x88, 0x3d, 0x90, 0x0f, 0xd1, 0x3e,
  0x20, 0x5b, 0x2a, 0x3f, 0x9b, 0x83, 0xe5, 0xbe, 0xd4, 0xa7, 0xd6, 0xbe,
  0xfd, 0x24, 0x23, 0xbf, 0x77, 0xb1, 0x5f, 0xc0, 0x95, 0x97, 0xdc, 0x3e,
  0x88, 0xff, 0x35, 0x3f, 0xb6, 0x3b, 0x52, 0x3f, 0xe4, 0x8c, 0xfb, 0x3e,
  0x6e, 0x77, 0x9a, 0x3f, 0xba, 0x2e, 0x4d, 0xbf, 0x47, 0x47, 0x12, 0xbf,
  0xbf, 0x20, 0xdc, 0xbd, 0x19, 0xa5, 0x0c, 0x3f, 0x00, 0x00, 0x00, 0x00,
  0xba, 0xfb, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00,
  0x02, 0x48, 0xf5, 0xbe, 0x24, 0xb8, 0x26, 0xbf, 0x70, 0x71, 0x37, 0x3d,
  0xa0, 0xb4, 0x54, 0xbe, 0xc6, 0x46, 0x2b, 0xbf, 0xc3, 0x8c, 0x17, 0x3f,
  0xfd, 0xfb, 0x5a, 0xbe, 0xec, 0x12, 0xc2, 0x3e, 0xfc, 0xb2, 0x0c, 0xbe,
  0x84, 0x22, 0xed, 0x3e, 0x48, 0xba, 0x4d, 0xbe, 0xe0, 0xb8, 0x33, 0xbe,
  0xe0, 0x63, 0x72, 0xbd, 0xc0, 0x20, 0xcb, 0x3e, 0x0c, 0x75, 0xfb, 0x3e,
  0xf0, 0x3b, 0x0b, 0xbd, 0x24, 0x12, 0x5b, 0xbe, 0x10, 0xd4, 0x3a, 0xbf,
  0x40, 0xaf, 0x8e, 0xbc, 0x00, 0xb4, 0x39, 0x3e, 0x00, 0x00, 0x00, 0x00,
  0x1a, 0xfc, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00,
  0x79, 0xdd, 0x27, 0x3f, 0x52, 0x40, 0x89, 0xbe, 0x9f, 0x5e, 0x62, 0x3e,
  0x67, 0x95, 0x24, 0x3c, 0x13, 0xef, 0x09, 0x40, 0xaa, 0xdf, 0x8f, 0x3f,
  0x40, 0xe3, 0xec, 0x3d, 0x2f, 0xa7, 0x20, 0x3f, 0x0e, 0x32, 0x8b, 0x3f,
  0x23, 0x8b, 0x5e, 0x3f, 0xee, 0x57, 0xf3, 0xbf, 0xa0, 0x31, 0x17, 0x3f,
  0x27, 0xe0, 0x17, 0x3f, 0x4a, 0x2a, 0xfd, 0x3d, 0x5a, 0xe2, 0x19, 0xbf,
  0x23, 0x32, 0xdc, 0xbf, 0x18, 0x9d, 0x5d, 0xbe, 0x6c, 0xa2, 0x35, 0x3f,
  0x94, 0x6f, 0x84, 0xbc, 0xb5, 0x43, 0x33, 0xbf, 0x00, 0x00, 0x00, 0x00,
  0x7a, 0xfc, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00,
  0x8c, 0xef, 0xc0, 0x3e, 0xe9, 0x0c, 0x02, 0x3f, 0x04, 0xe1, 0x4b, 0xbf,
  0x33, 0xfb, 0xd4, 0x3d, 0x59, 0x89, 0x4d, 0xbf, 0xae, 0x87, 0x90, 0x3e,
  0xa5, 0x83, 0x0a, 0x3f, 0x76, 0x24, 0x67, 0x3f, 0xd0, 0x1a, 0x03, 0xbf,
  0x5d, 0xc4, 0x2e, 0x3e, 0x7c, 0x6e, 0x17, 0x3e, 0x3c, 0x91, 0x89, 0x3e,
  0xdc, 0x89, 0xae, 0xbe, 0xe4, 0xdd, 0x3f, 0xbf, 0x70, 0xff, 0x69, 0x3d,
  0x90, 0xe9, 0x92, 0x3d, 0x77, 0x27, 0x1e, 0x3f, 0xf9, 0x13, 0x99, 0x3d,
  0xc6, 0x85, 0xc4, 0x3e, 0xae, 0x7d, 0xcf, 0x3e, 0x92, 0x12, 0x0f, 0x3f,
  0x78, 0xe8, 0x64, 0x3f, 0xe4, 0x87, 0x85, 0x3f, 0x07, 0xb4, 0x38, 0x3f,
  0x47, 0x58, 0x65, 0x3f, 0xea, 0xfc, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00,
  0x64, 0x00, 0x00, 0x00, 0x14, 0x39, 0xc4, 0x3e, 0x18, 0xac, 0x93, 0x3e,
  0x6b, 0xe5, 0xb7, 0x3e, 0xef, 0x5b, 0xa1, 0xbe, 0x16, 0x86, 0xbc, 0x3e,
  0xcf, 0x16, 0x42, 0xbf, 0x9b, 0x54, 0x3c, 0x3f, 0x20, 0xf3, 0xd2, 0x3e,
  0x92, 0x6a, 0x43, 0x3f, 0xa9, 0xcf, 0x2f, 0xbe, 0x28, 0xc3, 0x66, 0xbe,
  0x40, 0xc9, 0x92, 0xbd, 0x1e, 0x47, 0xb8, 0xbe, 0x2b, 0x6c, 0x19, 0xbf,
  0x60, 0x7d, 0x1e, 0x3e, 0xaa, 0x57, 0x1c, 0xbf, 0x98, 0xf7, 0x66, 0x3e,
  0x52, 0xbb, 0x3e, 0xbf, 0x00, 0x63, 0x89, 0x3e, 0x8c, 0x64, 0x9e, 0x3e,
  0x9d, 0xb6, 0xd0, 0xbe, 0xfe, 0xf1, 0xad, 0xbe, 0x73, 0xf6, 0x1a, 0xbf,
  0x85, 0x49, 0x3f, 0xbf, 0x90, 0x96, 0x95, 0x3e, 0x5a, 0xfd, 0xff, 0xff,
  0x04, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0xd6, 0xad, 0xd7, 0xbd,
  0xb0, 0xbb, 0x4f, 0x3f, 0xc7, 0x10, 0x39, 0x3f, 0x62, 0x74, 0x6c, 0x3d,
  0x0a, 0x26, 0x61, 0xbf, 0x70, 0xbe, 0x7b, 0xbd, 0xc1, 0x92, 0xc2, 0xbe,
  0x7e, 0x44, 0xde, 0xbe, 0xaa, 0xa6, 0x03, 0x3f, 0x1c, 0x11, 0x04, 0xbf,
  0xc2, 0x7e, 0xef, 0xbe, 0xa2, 0xdd, 0x49, 0x3f, 0xc1, 0x2c, 0xdf, 0x3e,
  0x3a, 0x02, 0xef, 0xbc, 0xe5, 0x48, 0xe0, 0xbd, 0x37, 0x08, 0xd7, 0x3e,
  0x3a, 0x00, 0x5d, 0x3f, 0xee, 0x6f, 0xbc, 0x3d, 0x44, 0x86, 0x48, 0x3f,
  0x07, 0x65, 0x0c, 0x3f, 0xb2, 0x6c, 0x6d, 0xbf, 0xc3, 0x18, 0x1c, 0x3d,
  0x77, 0xe5, 0x4a, 0x3e, 0x64, 0xce, 0x0f, 0x3e, 0xdf, 0x34, 0x82, 0x3f,
  0xca, 0xfd, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0xdc, 0x13, 0x0a, 0xbe, 0x4b, 0x8b, 0x86, 0xbc, 0x73, 0xcb, 0x18, 0x3e,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xea, 0xfd, 0xff, 0xff,
  0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x83, 0x1b, 0xe3, 0x3e,
  0x00, 0x00, 0x00, 0x00, 0xd8, 0x4f, 0x2f, 0xbe, 0xdd, 0xa0, 0x96, 0x3c,
  0x00, 0x00, 0x00, 0x00, 0x0a, 0xfe, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x5e, 0xb8, 0xd2, 0xbd,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x2a, 0xfe, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0xfe, 0x4e, 0xb2, 0xbd, 0xdc, 0x0e, 0x26, 0x3e, 0x29, 0x63, 0xd3, 0xbc,
  0x59, 0x5d, 0x5c, 0x3d, 0x00, 0x00, 0x00, 0x00, 0x4a, 0xfe, 0xff, 0xff,
  0x04, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x71, 0xb4, 0x9f, 0x3e,
  0x9f, 0x55, 0xc9, 0xbe, 0x00, 0x00, 0x00, 0x00, 0xdd, 0xe3, 0x91, 0xbe,
  0xe4, 0xa2, 0xa0, 0xbd, 0x6a, 0xfe, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0xc9, 0x78, 0xd6, 0xbd, 0xfc, 0x57, 0xc7, 0x3d,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x8a, 0xfe, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x1f, 0x85, 0x83, 0xbe, 0x00, 0x00, 0x00, 0x00, 0xed, 0x0a, 0xc2, 0x3d,
  0x1e, 0x0a, 0x37, 0x3e, 0x90, 0xc9, 0x36, 0xbe, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xf4, 0xf3, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0xf4, 0xff, 0xff,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x14, 0xf4, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x24, 0xf4, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x4d, 0x4c, 0x49, 0x52,
  0x20, 0x43, 0x6f, 0x6e, 0x76, 0x65, 0x72, 0x74, 0x65, 0x64, 0x2e, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00,
  0x18, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x14, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x80, 0x02, 0x00, 0x00, 0x6c, 0x02, 0x00, 0x00,
  0x60, 0x02, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00,
  0x09, 0x00, 0x00, 0x00, 0x08, 0x02, 0x00, 0x00, 0xc0, 0x01, 0x00, 0x00,
  0x78, 0x01, 0x00, 0x00, 0x40, 0x01, 0x00, 0x00, 0x08, 0x01, 0x00, 0x00,
  0xd0, 0x00, 0x00, 0x00, 0x8c, 0x00, 0x00, 0x00, 0x48, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x18, 0x00, 0x08, 0x00,
  0x0c, 0x00, 0x10, 0x00, 0x07, 0x00, 0x14, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x09, 0x02, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x8e, 0xff, 0xff, 0xff,
  0x00, 0x00, 0x80, 0x3f, 0x01, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0xae, 0xfe, 0xff, 0xff,
  0x00, 0x00, 0x00, 0x08, 0x18, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0xfc, 0xf4, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00,
  0x1a, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x07, 0x00, 0x14, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x01, 0x00, 0x00, 0x00,
  0x24, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x06, 0x00, 0x08, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x16, 0x00, 0x00, 0x00, 0x2e, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x08,
  0x1c, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0xda, 0xfe, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00,
  0x16, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00,
  0x0f, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x62, 0xff, 0xff, 0xff,
  0x00, 0x00, 0x00, 0x08, 0x1c, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x0e, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
  0x01, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x96, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x08, 0x1c, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x42, 0xff, 0xff, 0xff,
  0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x07, 0x00, 0x00, 0x00, 0xca, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x08,
  0x1c, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x76, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x0b, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x07, 0x00, 0x10, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x1c, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xba, 0xff, 0xff, 0xff,
  0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x16, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x0c, 0x00, 0x07, 0x00, 0x10, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x08, 0x24, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00,
  0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x08, 0x00, 0x07, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00,
  0x11, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x0a, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x19, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00,
  0xf8, 0x08, 0x00, 0x00, 0xa0, 0x08, 0x00, 0x00, 0x5c, 0x08, 0x00, 0x00,
  0xfc, 0x07, 0x00, 0x00, 0x8c, 0x07, 0x00, 0x00, 0x2c, 0x07, 0x00, 0x00,
  0xcc, 0x06, 0x00, 0x00, 0x6c, 0x06, 0x00, 0x00, 0x0c, 0x06, 0x00, 0x00,
  0xb0, 0x05, 0x00, 0x00, 0x68, 0x05, 0x00, 0x00, 0x20, 0x05, 0x00, 0x00,
  0xd8, 0x04, 0x00, 0x00, 0x90, 0x04, 0x00, 0x00, 0x48, 0x04, 0x00, 0x00,
  0x00, 0x04, 0x00, 0x00, 0xb8, 0x03, 0x00, 0x00, 0x44, 0x03, 0x00, 0x00,
  0xd0, 0x02, 0x00, 0x00, 0x5c, 0x02, 0x00, 0x00, 0xe8, 0x01, 0x00, 0x00,
  0x74, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0xa4, 0x00, 0x00, 0x00,
  0x4c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x84, 0xf7, 0xff, 0xff,
  0x34, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0xff, 0xff, 0xff, 0xff, 0x03, 0x00, 0x00, 0x00, 0x70, 0xf7, 0xff, 0xff,
  0x08, 0x00, 0x00, 0x00, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79,
  0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0xc8, 0xf7, 0xff, 0xff, 0x44, 0x00, 0x00, 0x00,
  0x19, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
  0x03, 0x00, 0x00, 0x00, 0xb4, 0xf7, 0xff, 0xff, 0x1b, 0x00, 0x00, 0x00,
  0x66, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31,
  0x2f, 0x73, 0x61, 0x6c, 0x69, 0x64, 0x61, 0x2f, 0x42, 0x69, 0x61, 0x73,
  0x41, 0x64, 0x64, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x1c, 0xf8, 0xff, 0xff, 0x48, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
  0x0c, 0x00, 0x00, 0x00, 0x08, 0xf8, 0xff, 0xff, 0x1f, 0x00, 0x00, 0x00,
  0x66, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31,
  0x2f, 0x63, 0x6f, 0x6e, 0x63, 0x61, 0x74, 0x65, 0x6e, 0x61, 0x74, 0x65,
  0x2f, 0x63, 0x6f, 0x6e, 0x63, 0x61, 0x74, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x74, 0xf8, 0xff, 0xff,
  0x60, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0xff, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x60, 0xf8, 0xff, 0xff,
  0x36, 0x00, 0x00, 0x00, 0x66, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e,
  0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x63, 0x61, 0x70, 0x61, 0x5f, 0x32, 0x5a,
  0x2f, 0x52, 0x65, 0x6c, 0x75, 0x3b, 0x66, 0x75, 0x6e, 0x63, 0x74, 0x69,
  0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x63, 0x61, 0x70, 0x61, 0x5f,
  0x32, 0x5a, 0x2f, 0x42, 0x69, 0x61, 0x73, 0x41, 0x64, 0x64, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0xe4, 0xf8, 0xff, 0xff, 0x60, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00,
  0x1c, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x05, 0x00, 0x00, 0x00,
  0xd0, 0xf8, 0xff, 0xff, 0x36, 0x00, 0x00, 0x00, 0x66, 0x75, 0x6e, 0x63,
  0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x63, 0x61, 0x70,
  0x61, 0x5f, 0x31, 0x5a, 0x2f, 0x52, 0x65, 0x6c, 0x75, 0x3b, 0x66, 0x75,
  0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x63,
  0x61, 0x70, 0x61, 0x5f, 0x31, 0x5a, 0x2f, 0x42, 0x69, 0x61, 0x73, 0x41,
  0x64, 0x64, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x00, 0x00, 0x54, 0xf9, 0xff, 0xff, 0x60, 0x00, 0x00, 0x00,
  0x15, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
  0x04, 0x00, 0x00, 0x00, 0x40, 0xf9, 0xff, 0xff, 0x36, 0x00, 0x00, 0x00,
  0x66, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31,
  0x2f, 0x63, 0x61, 0x70, 0x61, 0x5f, 0x32, 0x59, 0x2f, 0x52, 0x65, 0x6c,
  0x75, 0x3b, 0x66, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c,
  0x5f, 0x31, 0x2f, 0x63, 0x61, 0x70, 0x61, 0x5f, 0x32, 0x59, 0x2f, 0x42,
  0x69, 0x61, 0x73, 0x41, 0x64, 0x64, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xc4, 0xf9, 0xff, 0xff,
  0x60, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0xff, 0xff, 0xff, 0xff, 0x05, 0x00, 0x00, 0x00, 0xb0, 0xf9, 0xff, 0xff,
  0x36, 0x00, 0x00, 0x00, 0x66, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e,
  0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x63, 0x61, 0x70, 0x61, 0x5f, 0x31, 0x59,
  0x2f, 0x52, 0x65, 0x6c, 0x75, 0x3b, 0x66, 0x75, 0x6e, 0x63, 0x74, 0x69,
  0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x63, 0x61, 0x70, 0x61, 0x5f,
  0x31, 0x59, 0x2f, 0x42, 0x69, 0x61, 0x73, 0x41, 0x64, 0x64, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x34, 0xfa, 0xff, 0xff, 0x60, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x1c, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00,
  0x20, 0xfa, 0xff, 0xff, 0x36, 0x00, 0x00, 0x00, 0x66, 0x75, 0x6e, 0x63,
  0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x63, 0x61, 0x70,
  0x61, 0x5f, 0x32, 0x58, 0x2f, 0x52, 0x65, 0x6c, 0x75, 0x3b, 0x66, 0x75,
  0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x63,
  0x61, 0x70, 0x61, 0x5f, 0x32, 0x58, 0x2f, 0x42, 0x69, 0x61, 0x73, 0x41,
  0x64, 0x64, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0xa4, 0xfa, 0xff, 0xff, 0x60, 0x00, 0x00, 0x00,
  0x12, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
  0x05, 0x00, 0x00, 0x00, 0x90, 0xfa, 0xff, 0xff, 0x36, 0x00, 0x00, 0x00,
  0x66, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31,
  0x2f, 0x63, 0x61, 0x70, 0x61, 0x5f, 0x31, 0x58, 0x2f, 0x52, 0x65, 0x6c,
  0x75, 0x3b, 0x66, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c,
  0x5f, 0x31, 0x2f, 0x63, 0x61, 0x70, 0x61, 0x5f, 0x31, 0x58, 0x2f, 0x42,
  0x69, 0x61, 0x73, 0x41, 0x64, 0x64, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0xfe, 0xfb, 0xff, 0xff,
  0x34, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0xf0, 0xfa, 0xff, 0xff, 0x1a, 0x00, 0x00, 0x00,
  0x66, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31,
  0x2f, 0x73, 0x61, 0x6c, 0x69, 0x64, 0x61, 0x2f, 0x4d, 0x61, 0x74, 0x4d,
  0x75, 0x6c, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x0c, 0x00, 0x00, 0x00, 0x42, 0xfc, 0xff, 0xff, 0x34, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x34, 0xfb, 0xff, 0xff, 0x1b, 0x00, 0x00, 0x00, 0x66, 0x75, 0x6e, 0x63,
  0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x63, 0x61, 0x70,
  0x61, 0x5f, 0x32, 0x5a, 0x2f, 0x4d, 0x61, 0x74, 0x4d, 0x75, 0x6c, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x86, 0xfc, 0xff, 0xff, 0x34, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00,
  0x0c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x78, 0xfb, 0xff, 0xff,
  0x1b, 0x00, 0x00, 0x00, 0x66, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e,
  0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x63, 0x61, 0x70, 0x61, 0x5f, 0x32, 0x59,
  0x2f, 0x4d, 0x61, 0x74, 0x4d, 0x75, 0x6c, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0xca, 0xfc, 0xff, 0xff,
  0x34, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0xbc, 0xfb, 0xff, 0xff, 0x1b, 0x00, 0x00, 0x00,
  0x66, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31,
  0x2f, 0x63, 0x61, 0x70, 0x61, 0x5f, 0x32, 0x58, 0x2f, 0x4d, 0x61, 0x74,
  0x4d, 0x75, 0x6c, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x00, 0x00, 0x0e, 0xfd, 0xff, 0xff, 0x34, 0x00, 0x00, 0x00,
  0x0d, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x00, 0xfc, 0xff, 0xff, 0x1b, 0x00, 0x00, 0x00, 0x66, 0x75, 0x6e, 0x63,
  0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x63, 0x61, 0x70,
  0x61, 0x5f, 0x31, 0x5a, 0x2f, 0x4d, 0x61, 0x74, 0x4d, 0x75, 0x6c, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x52, 0xfd, 0xff, 0xff, 0x34, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x0c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x44, 0xfc, 0xff, 0xff,
  0x1b, 0x00, 0x00, 0x00, 0x66, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e,
  0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x63, 0x61, 0x70, 0x61, 0x5f, 0x31, 0x59,
  0x2f, 0x4d, 0x61, 0x74, 0x4d, 0x75, 0x6c, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x96, 0xfd, 0xff, 0xff,
  0x34, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x88, 0xfc, 0xff, 0xff, 0x1b, 0x00, 0x00, 0x00,
  0x66, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31,
  0x2f, 0x63, 0x61, 0x70, 0x61, 0x5f, 0x31, 0x58, 0x2f, 0x4d, 0x61, 0x74,
  0x4d, 0x75, 0x6c, 0x00, 0x02, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x00, 0x00, 0xda, 0xfd, 0xff, 0xff, 0x4c, 0x00, 0x00, 0x00,
  0x0a, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0xcc, 0xfc, 0xff, 0xff, 0x33, 0x00, 0x00, 0x00, 0x66, 0x75, 0x6e, 0x63,
  0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x73, 0x61, 0x6c,
  0x69, 0x64, 0x61, 0x2f, 0x42, 0x69, 0x61, 0x73, 0x41, 0x64, 0x64, 0x2f,
  0x52, 0x65, 0x61, 0x64, 0x56, 0x61, 0x72, 0x69, 0x61, 0x62, 0x6c, 0x65,
  0x4f, 0x70, 0x2f, 0x72, 0x65, 0x73, 0x6f, 0x75, 0x72, 0x63, 0x65, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x32, 0xfe, 0xff, 0xff,
  0x50, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x24, 0xfd, 0xff, 0xff, 0x34, 0x00, 0x00, 0x00,
  0x66, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31,
  0x2f, 0x63, 0x61, 0x70, 0x61, 0x5f, 0x32, 0x5a, 0x2f, 0x42, 0x69, 0x61,
  0x73, 0x41, 0x64, 0x64, 0x2f, 0x52, 0x65, 0x61, 0x64, 0x56, 0x61, 0x72,
  0x69, 0x61, 0x62, 0x6c, 0x65, 0x4f, 0x70, 0x2f, 0x72, 0x65, 0x73, 0x6f,
  0x75, 0x72, 0x63, 0x65, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x8e, 0xfe, 0xff, 0xff, 0x50, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x80, 0xfd, 0xff, 0xff, 0x34, 0x00, 0x00, 0x00, 0x66, 0x75, 0x6e, 0x63,
  0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x63, 0x61, 0x70,
  0x61, 0x5f, 0x32, 0x59, 0x2f, 0x42, 0x69, 0x61, 0x73, 0x41, 0x64, 0x64,
  0x2f, 0x52, 0x65, 0x61, 0x64, 0x56, 0x61, 0x72, 0x69, 0x61, 0x62, 0x6c,
  0x65, 0x4f, 0x70, 0x2f, 0x72, 0x65, 0x73, 0x6f, 0x75, 0x72, 0x63, 0x65,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0xea, 0xfe, 0xff, 0xff, 0x50, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
  0x0c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xdc, 0xfd, 0xff, 0xff,
  0x34, 0x00, 0x00, 0x00, 0x66, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e,
  0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x63, 0x61, 0x70, 0x61, 0x5f, 0x32, 0x58,
  0x2f, 0x42, 0x69, 0x61, 0x73, 0x41, 0x64, 0x64, 0x2f, 0x52, 0x65, 0x61,
  0x64, 0x56, 0x61, 0x72, 0x69, 0x61, 0x62, 0x6c, 0x65, 0x4f, 0x70, 0x2f,
  0x72, 0x65, 0x73, 0x6f, 0x75, 0x72, 0x63, 0x65, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x46, 0xff, 0xff, 0xff,
  0x50, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x38, 0xfe, 0xff, 0xff, 0x34, 0x00, 0x00, 0x00,
  0x66, 0x75, 0x6e, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31,
  0x2f, 0x63, 0x61, 0x70, 0x61, 0x5f, 0x31, 0x5a, 0x2f, 0x42, 0x69, 0x61,
  0x73, 0x41, 0x64, 0x64, 0x2f, 0x52, 0x65, 0x61, 0x64, 0x56, 0x61, 0x72,
  0x69, 0x61, 0x62, 0x6c, 0x65, 0x4f, 0x70, 0x2f, 0x72, 0x65, 0x73, 0x6f,
  0x75, 0x72, 0x63, 0x65, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x00, 0x00, 0xa2, 0xff, 0xff, 0xff, 0x50, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x94, 0xfe, 0xff, 0xff, 0x34, 0x00, 0x00, 0x00, 0x66, 0x75, 0x6e, 0x63,
  0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x63, 0x61, 0x70,
  0x61, 0x5f, 0x31, 0x59, 0x2f, 0x42, 0x69, 0x61, 0x73, 0x41, 0x64, 0x64,
  0x2f, 0x52, 0x65, 0x61, 0x64, 0x56, 0x61, 0x72, 0x69, 0x61, 0x62, 0x6c,
  0x65, 0x4f, 0x70, 0x2f, 0x72, 0x65, 0x73, 0x6f, 0x75, 0x72, 0x63, 0x65,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x0e, 0x00, 0x14, 0x00, 0x04, 0x00, 0x00, 0x00, 0x08, 0x00,
  0x0c, 0x00, 0x10, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x00, 0xff, 0xff, 0xff, 0x34, 0x00, 0x00, 0x00, 0x66, 0x75, 0x6e, 0x63,
  0x74, 0x69, 0x6f, 0x6e, 0x61, 0x6c, 0x5f, 0x31, 0x2f, 0x63, 0x61, 0x70,
  0x61, 0x5f, 0x31, 0x58, 0x2f, 0x42, 0x69, 0x61, 0x73, 0x41, 0x64, 0x64,
  0x2f, 0x52, 0x65, 0x61, 0x64, 0x56, 0x61, 0x72, 0x69, 0x61, 0x62, 0x6c,
  0x65, 0x4f, 0x70, 0x2f, 0x72, 0x65, 0x73, 0x6f, 0x75, 0x72, 0x63, 0x65,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x80, 0xff, 0xff, 0xff, 0x30, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x1c, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x05, 0x00, 0x00, 0x00,
  0x6c, 0xff, 0xff, 0xff, 0x07, 0x00, 0x00, 0x00, 0x69, 0x6e, 0x70, 0x75,
  0x74, 0x5f, 0x33, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x00, 0x00, 0xc0, 0xff, 0xff, 0xff, 0x30, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
  0x05, 0x00, 0x00, 0x00, 0xac, 0xff, 0xff, 0xff, 0x07, 0x00, 0x00, 0x00,
  0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x32, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x14, 0x00, 0x18, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x14, 0x00, 0x14, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
  0x05, 0x00, 0x00, 0x00, 0x04, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x07, 0x00, 0x00, 0x00, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x31, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0xfa, 0xff, 0xff, 0xff, 0x00, 0x19, 0x06, 0x00,
  0x06, 0x00, 0x05, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x02, 0x0a, 0x00,
  0x0c, 0x00, 0x07, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0a, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x09, 0x03, 0x00, 0x00, 0x00
};
const int exercisemodel_len = 4592;

float x_test[15][3][5] = {{{0.33333333, 0.35686275, 0.50588235, 0.51372549, 0.50588235}, {0.47058824, 0.4745098,  0.49411765, 0.49019608, 0.49019608} , {0.40784314, 0.44313725, 0.48627451, 0.49803922, 0.49411765}}, 
                        {{0.30588235, 0.30196078, 0.35294118, 0.38039216, 0.4745098}  , {0.4745098,  0.48235294, 0.48235294, 0.48627451, 0.49019608} , {0.35294118, 0.35686275, 0.40784314, 0.44705882, 0.48627451}}, 
                        {{0.35686275, 0.39215686, 0.52156863, 0.50588235, 0.50980392} , {0.47058824, 0.48627451, 0.49411765, 0.49019608, 0.48627451} , {0.42352941, 0.46666667, 0.48627451, 0.49803922, 0.49411765}},
                        {{0.32941176, 0.33333333, 0.37254902, 0.43137255, 0.45882353} , {0.4745098,  0.4745098 , 0.47843137, 0.48235294, 0.48627451} , {0.43921569, 0.44313725, 0.45882353, 0.48235294, 0.48627451}}, 
                        {{0.34901961, 0.38039216, 0.44313725, 0.45882353, 0.43529412} , {0.47058824, 0.4745098,  0.48235294, 0.48235294, 0.47843137} , {0.44313725, 0.46666667, 0.48627451, 0.49019608, 0.48627451}},
                        {{0.34509804, 0.35686275, 0.42352941, 0.42352941, 0.36470588} , {0.49803922, 0.49803922, 0.49019608, 0.49411765, 0.49803922} , {0.50588235, 0.50980392, 0.49411765, 0.49803922, 0.50196078}}, 
                        {{0.41960784, 0.36862745, 0.32156863, 0.31372549, 0.35294118} , {0.47058824, 0.46666667, 0.4627451 , 0.45882353, 0.46666667} , {0.48627451, 0.46666667, 0.45490196, 0.45098039, 0.46666667}}, 
                        {{0.33333333, 0.2627451 , 0.33333333, 0.36470588, 0.45490196} , {0.48627451, 0.50196078, 0.48235294, 0.48235294, 0.47843137} , {0.37647059, 0.3254902 , 0.37647059, 0.41568627, 0.47843137}}, 
                        {{0.35294118, 0.4       , 0.50980392, 0.50588235, 0.50980392} , {0.48627451, 0.4745098 , 0.48235294, 0.49411765, 0.49019608} , {0.4       , 0.45098039, 0.48235294, 0.49411765, 0.49411765}}, 
                        {{0.45098039, 0.53333333, 0.50588235, 0.48235294, 0.35294118} , {0.49411765, 0.48627451, 0.48627451, 0.48235294, 0.47843137} , {0.47843137, 0.49019608, 0.49019608, 0.49411765, 0.42352941}},
                        {{0.48627451, 0.40392157, 0.31372549, 0.28235294, 0.33333333} , {0.49019608, 0.47843137, 0.47843137, 0.43137255, 0.49019608} , {0.48627451, 0.43529412, 0.36470588, 0.29803922, 0.37254902}}, 
                        {{0.35294118, 0.31764706, 0.34117647, 0.38431373, 0.44705882} , {0.48235294, 0.48627451, 0.48627451, 0.48627451, 0.48627451} , {0.45882353, 0.44705882, 0.45490196, 0.4745098 , 0.48627451}}, 
                        {{0.41176471, 0.40392157, 0.37254902, 0.36862745, 0.39215686} , {0.49803922, 0.50196078, 0.50980392, 0.50588235, 0.50196078} , {0.50196078, 0.50196078, 0.50980392, 0.50980392, 0.50588235}}, 
                        {{0.40392157, 0.37647059, 0.34117647, 0.3254902 , 0.36078431} , {0.48235294, 0.48235294, 0.47843137, 0.47843137, 0.48235294} , {0.47843137, 0.45882353, 0.44313725, 0.43529412, 0.45490196}}, 
                        {{0.35294118, 0.35686275, 0.41568627, 0.41176471, 0.39215686} , {0.4745098 , 0.4745098 , 0.4745098 , 0.47843137, 0.47058824} , {0.50588235, 0.50588235, 0.49411765, 0.49411765, 0.50196078}}, 
                        
                        };

float labels[15][3] = {{0.,0.,1.},
                      {0.,0.,1.},
                      {0.,0.,1.},
                      {0.,1.,0.},
                      {0.,1.,0.},
                      {1.,0.,0.},
                      {0.,1.,0.},
                      {0.,0.,1.},
                      {0.,0.,1.},
                      {0.,0.,1.},
                      {0.,0.,1.},
                      {0.,1.,0.},
                      {1.,0.,0.},
                      {0.,1.,0.},
                      {1.,0.,0.},
                      };
