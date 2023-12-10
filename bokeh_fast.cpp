#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <tuple>

double gaussian_factor(int radius, double dist) {
    double sigma = 0.4;
    if (radius == 0) {
        return 1.0;
    }
    double fraction = dist / radius;
    double power = (-1.0 / 2) * std::pow((fraction / sigma), 2);
    double value = (1.0 / (sigma * 2.5066282746310002)) * std::exp(power);
    return value;
}

double euclidean_distance(std::pair<int, int> p1, std::pair<int, int> p2) {
    return std::sqrt(std::pow(p2.first - p1.first, 2) + std::pow(p2.second - p1.second, 2));
}


double calculate_coc_radius(double depth_diff, double focus_depth, double size_value, double coc_scale) {
    double max_radius = size_value / coc_scale;
    double depth_diff_percent = depth_diff / focus_depth;
    double temp_val = depth_diff_percent * 12 - 6;
    double scale = 1.0 / (1.0 + std::exp(-temp_val));
    double radius = scale * max_radius;
    return radius;
}

bool is_in_bokeh_shape_euclidean(std::pair<int, int> p1, std::pair<int, int> p2, int width, int height, double radius, double dist) {
    if (p2.first < 0 || p2.first >= width || p2.second < 0 || p2.second >= height) {
        return false;
    }
    return dist <= radius;
}

bool is_in_image(std::pair<int, int> p1, std::pair<int, int> p2, int width, int height, double radius) {
    if (p2.first < 0 || p2.first >= width || p2.second < 0 || p2.second >= height) {
        return false;
    }
    return true;
}

std::vector<std::vector<std::tuple<int, int, int>>> bokeh(std::vector<std::vector<std::tuple<int, int, int>>> &image, std::vector<std::vector<std::tuple<int, int, int>>> &depth_map, int width, int height, double focus_depth, double max_depth_diff, double coc_scale) {
    int dim_x = width;
    int dim_y = height;
    std::vector<std::vector<std::tuple<int, int, int>>> new_image(dim_x, std::vector<std::tuple<int, int, int>>(dim_y, std::make_tuple(0, 0, 0)));
    std::vector<std::vector<double>> new_image_weights(dim_x, std::vector<double>(dim_y, 0.0));

    for (int x = 0; x < dim_x; x++) {
        for (int y = 0; y < dim_y; y++) {
            double circle_of_confusion_radius = std::abs(std::round(calculate_coc_radius(std::abs(focus_depth - std::get<0>(depth_map[x][y])), focus_depth, (dim_x + dim_y) / 2, coc_scale)));
            for (int i = x - circle_of_confusion_radius; i <= x + circle_of_confusion_radius; i++) {
                for (int j = y - circle_of_confusion_radius; j <= y + circle_of_confusion_radius; j++) {
                    double dist = euclidean_distance(std::make_pair(x, y), std::make_pair(i, j));
                    if (is_in_bokeh_shape_euclidean(std::make_pair(x, y), std::make_pair(i, j), width, height, circle_of_confusion_radius, dist)) {
                        if (std::get<0>(depth_map[x][y]) >= std::get<0>(depth_map[i][j]) && std::abs(std::get<0>(depth_map[x][y]) - std::get<0>(depth_map[i][j])) < max_depth_diff) {
                            double g_factor = gaussian_factor(circle_of_confusion_radius, dist);
                            std::get<0>(new_image[i][j]) += g_factor * std::get<0>(image[x][y]);
                            std::get<1>(new_image[i][j]) += g_factor * std::get<1>(image[x][y]);
                            std::get<2>(new_image[i][j]) += g_factor * std::get<2>(image[x][y]);
                            new_image_weights[i][j] += g_factor;
                        }
                    }
                }
            }
        }
    }

    std::vector<std::vector<std::tuple<int, int, int>>> new_image_tuples(dim_x, std::vector<std::tuple<int, int, int>>(dim_y));
    for (int a = 0; a < dim_x; ++a) {
        for (int b = 0; b < dim_y; ++b) {
            double w = new_image_weights[a][b];
            std::tuple<int, int, int> t = std::make_tuple(static_cast<int>(std::round(std::get<0>(new_image[a][b]) / w)),
                                                          static_cast<int>(std::round(std::get<1>(new_image[a][b]) / w)),
                                                          static_cast<int>(std::round(std::get<2>(new_image[a][b]) / w)));
            new_image_tuples[a][b] = t;
        }
    }
    return new_image_tuples;
}

void writeImageToFile(const std::vector<std::vector<std::tuple<int, int, int>>> &image, const std::string &filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file." << std::endl;
        return;
    }

    for (const auto &row : image) {
        for (const auto &pixel : row) {
            outputFile << std::get<0>(pixel) << " " << std::get<1>(pixel) << " " << std::get<2>(pixel) << " ";
        }
    }
    outputFile << "\n";

    outputFile.close();
}


int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <width> <height> <point_x> <point_y> <image_array> <depth_map_array>" << std::endl;
        return 1;
    }

    int width = std::stoi(argv[1]);
    std::cout << "test";
    int height = std::stoi(argv[2]);
    double focus_depth = std::stof(argv[3]);
    double max_depth_diff = std::stof(argv[4]);
    double coc_scale = std::stof(argv[5]);


    std::ifstream file("image_array.txt");
    if (!file.is_open()) {
        std::cout << "Error opening file." << std::endl;
        return 1;
    }

    // Read image array from file
    std::string image_array_str;
    std::getline(file, image_array_str);
    std::vector<int> image_array;
    std::istringstream image_array_stream(image_array_str);
    int value;
    while (image_array_stream >> value) {
        image_array.push_back(value);
        if (image_array_stream.peek() == ' ') {
            image_array_stream.ignore();
        }
    }

    std::ifstream file2("depth_array.txt");
    if (!file2.is_open()) {
        std::cout << "Error opening file." << std::endl;
        return 1;
    }
    std::cout << "test2";


    // Read depth map array from file
    std::string depth_map_array_str;
    std::getline(file2, depth_map_array_str);
    std::vector<int> depth_map_array;
    std::istringstream depth_map_array_stream(depth_map_array_str);
    while (depth_map_array_stream >> value) {
        depth_map_array.push_back(value);
        if (depth_map_array_stream.peek() == ' ') {
            depth_map_array_stream.ignore();
        }
    }

    std::cout << "test3";


    std::vector<std::vector<std::tuple<int, int, int>>> new_image(width, std::vector<std::tuple<int, int, int>>(height, std::make_tuple(0, 0, 0)));

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int ind = (i * height * 3) + j * 3;
            int red = image_array[ind];
            int green = image_array[ind + 1];
            int blue = image_array[ind + 2];
            std::tuple<int, int, int> t = std::make_tuple(red, green, blue);
            new_image[i][j] = t;
        }
    }

    std::cout << "test4";

    std::vector<std::vector<std::tuple<int, int, int>>> new_image_depth(width, std::vector<std::tuple<int, int, int>>(height, std::make_tuple(0, 0, 0)));

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int ind = (i * height * 3) + j * 3;
            int red = depth_map_array[ind];
            int green = depth_map_array[ind + 1];
            int blue = depth_map_array[ind + 2];
            std::tuple<int, int, int> t = std::make_tuple(red, red, red);
            new_image_depth[i][j] = t;
        }
    }

    std::vector<std::vector<std::tuple<int, int, int>>> output = bokeh(new_image, new_image_depth, width, height, focus_depth, max_depth_diff, coc_scale);
    writeImageToFile(output, "output.txt");

    return 0;
}