#include "../Output/Picture.hpp"
#include <iostream>
#include <sstream>
#include <vector>
#include <filesystem>

void flipImages(const std::string& path) {
	for (const auto& entry : std::filesystem::directory_iterator(path)) {
		std::string filename = entry.path().string();
		if (entry.is_directory()) {
			flipImages(filename);
		}
		else if (filename.substr(filename.length() - 4, 4) == ".bmp") {
			static constexpr int ROWS = 2;
			static constexpr int COLS = 3;
			Picture oldImage;
			oldImage.load(filename, false);
			int width = oldImage.getWidth() / COLS;
			int height = oldImage.getHeight() / ROWS;
			Picture newImage(width * COLS, height * ROWS);
			for (int row = 0; row < ROWS; ++row) {
				for (int col = 0; col < COLS; ++col) {
					for (int x = 0; x < width; ++x) {
						for (int y = 0; y < height; ++y) {
							bool in_borders = 0 < y && 0 < x && y < height - 1 && x < width - 1;
							int src_x = col * width + x;
							int src_y = row * height + y;
							int dst_x = in_borders ? col * width + (width - x - 1) : src_x;
							int dst_y = src_y;
							if (row == (ROWS - 1) && in_borders)
							{
								dst_y = row * height + (height - y - 1);
							}
							newImage.setColor(dst_x, dst_y, oldImage.getColor(src_x, src_y));
						}
					}
				}
			}
			newImage.save(filename, false);
			std::cout << "Done flipping: " << filename << std::endl;
		}
	}
}

int main( int argc, char** argv )
{
	std::string directory = "results_0.500000";
	flipImages(directory);
	return 0;
}
