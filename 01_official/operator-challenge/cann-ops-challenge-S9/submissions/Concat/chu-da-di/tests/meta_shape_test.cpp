#define S9_METADATA_TEST
#include "../op_host/infer_shape.cpp"
#include <cstdlib>
#include <iostream>

static void Check(int id, const s9merge::Shapes& inputs, const s9merge::Shape& attrs,
                  bool valid, const s9merge::Shape& expected)
{
    s9merge::Shape actual;
    const bool accepted = s9merge::Infer(inputs, attrs, actual);
    if (accepted != valid || (accepted && actual != expected)) {
        std::cerr << "Metadata case " << id << " failed" << std::endl;
        std::exit(1);
    }
}
int main()
{
    Check(0, {{5,19},{5,7},{5,33}}, {1}, true, {5,59});
    Check(1, {{5,19},{5,7}}, {-1}, true, {5,26});
    Check(2, {{5,-1},{5,3}}, {1}, true, {5,-1});
    Check(3, {{},{}}, {0}, false, {});
    Check(4, {{2,3},{4,3}}, {1}, false, {});
    Check(5, {{2,3}}, {2}, false, {});
    std::cout << "Concat metadata: 6 checks passed" << std::endl;
    return 0;
}
