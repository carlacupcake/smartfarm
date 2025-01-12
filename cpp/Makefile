# Compiler
CXX = clang++

# Flags
CXXFLAGS = \
	-std=c++17 \
	-I./include \
	-I./libs/json/include \
	-Wno-deprecated-declarations

# Sources and target
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SOURCES))
TARGET = simulation

# Rules to build the simulation
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Rule to clean up the output files
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

