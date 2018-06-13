#ifndef CXXLINEMOD_H
#define CXXLINEMOD_H
#include <opencv2/core/core.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class poseRefine{
public:
    poseRefine(): residual(-1){}
    void process(cv::Mat& sceneDepth, cv::Mat& modelDepth, cv::Mat& sceneK, cv::Mat& modelK,
                    cv::Mat& modelR, cv::Mat& modelT, int detectX, int detectY);
    float getResidual();
    cv::Mat getR();
    cv::Mat getT();
private:
    cv::Mat R_refined, t_refiend;
    float residual;
};

namespace linemodLevelup {

struct Feature {
    int x;
    int y;
    int label;

    void read(const cv::FileNode& fn);
    void write(cv::FileStorage& fs) const;

    Feature() : x(0), y(0), label(0) {}
    Feature(int x, int y, int label);
};
inline Feature::Feature(int _x, int _y, int _label) : x(_x), y(_y), label(_label) {}

struct Template
{
    int width;
    int height;
    int pyramid_level;
    std::vector<Feature> features;

    void read(const cv::FileNode& fn);
    void write(cv::FileStorage& fs) const;
};

/**
 * \brief Represents a modality operating over an image pyramid.
 */
class QuantizedPyramid
{
public:
  // Virtual destructor
  virtual ~QuantizedPyramid() {}

  /**
   * \brief Compute quantized image at current pyramid level for online detection.
   *
   * \param[out] dst The destination 8-bit image. For each pixel at most one bit is set,
   *                 representing its classification.
   */
  virtual void quantize(cv::Mat& dst) const =0;
  /**
   * \brief Extract most discriminant features at current pyramid level to form a new template.
   *
   * \param[out] templ The new template.
   */
  virtual bool extractTemplate(Template& templ) const =0;

  /**
   * \brief Go to the next pyramid level.
   *
   * \todo Allow pyramid scale factor other than 2
   */
  virtual void pyrDown() =0;

protected:
  /// Candidate feature with a score
  struct Candidate
  {
    Candidate(int x, int y, int label, float score);

    /// Sort candidates with high score to the front
    bool operator<(const Candidate& rhs) const
    {
      return score > rhs.score;
    }

    Feature f;
    float score;
  };
  /**
   * \brief Choose candidate features so that they are not bunched together.
   *
   * \param[in]  candidates   Candidate features sorted by score.
   * \param[out] features     Destination vector of selected features.
   * \param[in]  num_features Number of candidates to select.
   * \param[in]  distance     Hint for desired distance between features.
   */
  static void selectScatteredFeatures(const std::vector<Candidate>& candidates,
                                      std::vector<Feature>& features,
                                      size_t num_features, float distance);
};

inline QuantizedPyramid::Candidate::Candidate(int x, int y, int label, float _score) : f(x, y, label), score(_score) {}

/**
 * \brief Interface for modalities that plug into the LINE template matching representation.
 *
 * \todo Max response, to allow optimization of summing (255/MAX) features as uint8
 */
class Modality
{
public:
  // Virtual destructor
  virtual ~Modality() {}

  /**
   * \brief Form a quantized image pyramid from a source image.
   *
   * \param[in] src  The source image. Type depends on the modality.
   * \param[in] mask Optional mask. If not empty, unmasked pixels are set to zero
   *                 in quantized image and cannot be extracted as features.
   */
  cv::Ptr<QuantizedPyramid> process(const std::vector<cv::Mat> &src,
                    const cv::Mat& mask = cv::Mat()) const
  {
    return processImpl(src, mask);
  }

  virtual std::string name() const =0;
  virtual void read(const cv::FileNode& fn) =0;
  virtual void write(cv::FileStorage& fs) const =0;

  /**
   * \brief Create modality by name.
   *
   * The following modality types are supported:
   * - "ColorGradient"
   * - "DepthNormal"
   */
  static cv::Ptr<Modality> create(const std::string& modality_type);

protected:
  // Indirection is because process() has a default parameter.
  virtual cv::Ptr<QuantizedPyramid> processImpl(const std::vector<cv::Mat> &src,
                        const cv::Mat& mask) const =0;
};

/**
 * \brief Modality that computes quantized gradient orientations from a color image.
 */
class ColorGradient : public Modality
{
public:
  /**
   * \brief Default constructor. Uses reasonable default parameter values.
   */
  ColorGradient();

  /**
   * \brief Constructor.
   *
   * \param weak_threshold   When quantizing, discard gradients with magnitude less than this.
   * \param num_features     How many features a template must contain.
   * \param strong_threshold Consider as candidate features only gradients whose norms are
   *                         larger than this.
   */
  ColorGradient(float weak_threshold, size_t num_features, float strong_threshold);

  virtual std::string name() const;

  float weak_threshold;
  size_t num_features;
  float strong_threshold;
  virtual void read(const cv::FileNode& fn);
  virtual void write(cv::FileStorage& fs) const;
protected:
  virtual cv::Ptr<QuantizedPyramid> processImpl(const std::vector<cv::Mat> &src,
                        const cv::Mat& mask) const;
};

/**
 * \brief Modality that computes quantized surface normals from a dense depth map.
 */
class DepthNormal : public Modality
{
public:
  /**
   * \brief Default constructor. Uses reasonable default parameter values.
   */
  DepthNormal();

  /**
   * \brief Constructor.
   *
   * \param distance_threshold   Ignore pixels beyond this distance.
   * \param difference_threshold When computing normals, ignore contributions of pixels whose
   *                             depth difference with the central pixel is above this threshold.
   * \param num_features         How many features a template must contain.
   * \param extract_threshold    Consider as candidate feature only if there are no differing
   *                             orientations within a distance of extract_threshold.
   */
  DepthNormal(int distance_threshold, int difference_threshold, size_t num_features,
              int extract_threshold);

  virtual std::string name() const;

  int distance_threshold;
  int difference_threshold;
  size_t num_features;
  int extract_threshold;

  virtual void read(const cv::FileNode& fn);
  virtual void write(cv::FileStorage& fs) const;

protected:
  virtual cv::Ptr<QuantizedPyramid> processImpl(const std::vector<cv::Mat> &src,
                        const cv::Mat& mask) const;
};

/**
 * \brief Represents a successful template match.
 */
struct Match
{
  Match()
  {
  }

  Match(int x, int y, float similarity, const std::string& class_id, int template_id);

  /// Sort matches with high similarity to the front
  bool operator<(const Match& rhs) const
  {
    // Secondarily sort on template_id for the sake of duplicate removal
    if (similarity != rhs.similarity)
      return similarity > rhs.similarity;
    else
      return template_id < rhs.template_id;
  }

  bool operator==(const Match& rhs) const
  {
    return x == rhs.x && y == rhs.y && similarity == rhs.similarity && class_id == rhs.class_id;
  }

  int x;
  int y;
  float similarity;
  std::string class_id;
  int template_id;
};

inline
Match::Match(int _x, int _y, float _similarity, const std::string& _class_id, int _template_id)
    : x(_x), y(_y), similarity(_similarity), class_id(_class_id), template_id(_template_id)
{}

/**
 * \brief Object detector using the LINE template matching algorithm with any set of
 * modalities.
 */
class Detector
{
public:
  /**
   * \brief Empty constructor, initialize with read().
   */
  Detector();

  Detector(std::vector<int> T);

  /**
   * \brief Constructor.
   *
   * \param modalities       Modalities to use (color gradients, depth normals, ...).
   * \param T_pyramid        Value of the sampling step T at each pyramid level. The
   *                         number of pyramid levels is T_pyramid.size().
   */
  Detector(const std::vector< cv::Ptr<Modality> >& modalities, const std::vector<int>& T_pyramid);

  /**
   * \brief Detect objects by template matching.
   *
   * Matches globally at the lowest pyramid level, then refines locally stepping up the pyramid.
   *
   * \param      sources   Source images, one for each modality.
   * \param      threshold Similarity threshold, a percentage between 0 and 100.
   * \param[out] matches   Template matches, sorted by similarity score.
   * \param      class_ids If non-empty, only search for the desired object classes.
   * \param[out] quantized_images Optionally return vector<Mat> of quantized images.
   * \param      masks     The masks for consideration during matching. The masks should be CV_8UC1
   *                       where 255 represents a valid pixel.  If non-empty, the vector must be
   *                       the same size as sources.  Each element must be
   *                       empty or the same size as its corresponding source.
   */
  std::vector<Match> match(const std::vector<cv::Mat>& sources, float threshold,
             const std::vector<std::string>& class_ids = std::vector<std::string>(),
                           const std::vector<cv::Mat>& masks = std::vector<cv::Mat>()) const;
  /**
   * \brief Add new object template.
   *
   * \param      sources      Source images, one for each modality.
   * \param      class_id     Object class ID.
   * \param      object_mask  Mask separating object from background.
   * \param[out] bounding_box Optionally return bounding box of the extracted features.
   *
   * \return Template ID, or -1 if failed to extract a valid template.
   */
  int addTemplate(const std::vector<cv::Mat>& sources, const std::string& class_id,
          const cv::Mat& object_mask);

  /**
   * \brief Get the modalities used by this detector.
   *
   * You are not permitted to add/remove modalities, but you may dynamic_cast them to
   * tweak parameters.
   */
  const std::vector< cv::Ptr<Modality> >& getModalities() const { return modalities; }

  /**
   * \brief Get sampling step T at pyramid_level.
   */
  int getT(int pyramid_level) const { return T_at_level[pyramid_level]; }

  /**
   * \brief Get number of pyramid levels used by this detector.
   */
  int pyramidLevels() const { return pyramid_levels; }

  /**
   * \brief Get the template pyramid identified by template_id.
   *
   * For example, with 2 modalities (Gradient, Normal) and two pyramid levels
   * (L0, L1), the order is (GradientL0, NormalL0, GradientL1, NormalL1).
   */
  const std::vector<Template>& getTemplates(const std::string& class_id, int template_id) const;

  int numTemplates() const;
  int numTemplates(const std::string& class_id) const;
  int numClasses() const { return static_cast<int>(class_templates.size()); }

  std::vector<std::string> classIds() const;

  void read(const cv::FileNode& fn);
  void write(cv::FileStorage& fs) const;

  std::string readClass(const cv::FileNode& fn, const std::string &class_id_override = "");
  void writeClass(const std::string& class_id, cv::FileStorage& fs) const;

  void readClasses(const std::vector<std::string>& class_ids,
                           const std::string& format = "templates_%s.yml.gz");
  void writeClasses(const std::string& format = "templates_%s.yml.gz") const;
protected:
  std::vector< cv::Ptr<Modality> > modalities;
  int pyramid_levels;
  std::vector<int> T_at_level;

  typedef std::vector<Template> TemplatePyramid;
  typedef std::map<std::string, std::vector<TemplatePyramid> > TemplatesMap;
  TemplatesMap class_templates;

  typedef std::vector<cv::Mat> LinearMemories;
  // Indexed as [pyramid level][modality][quantized label]
  typedef std::vector< std::vector<LinearMemories> > LinearMemoryPyramid;

  void matchClass(const LinearMemoryPyramid& lm_pyramid,
                  const std::vector<cv::Size>& sizes,
                  float threshold, std::vector<Match>& matches,
                  const std::string& class_id,
                  const std::vector<TemplatePyramid>& template_pyramids) const;

};

/**
 * \brief Factory function for detector using LINE-MOD algorithm with color gradients
 * and depth normals.
 *
 * Default parameter settings suitable for VGA images.
 */
cv::Ptr<linemodLevelup::Detector> getDefaultLINEMOD();
}

#endif
